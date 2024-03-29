// Author: Conrad Fernandez
// Student Number: 20219637

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE_WIDTH 16
#define TILE_HEIGHT 9
#define TILE_SIZE 16

void rand_matrix(float* matrix, int rows, int cols) {
	// Fill a rows x cols matrix with random floats
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			float randomNumber = 1 + (float)rand() / ((float)RAND_MAX / (255 - 1));
			matrix[i * cols + j] = randomNumber;
		}
	}
}

void print_matrix(float* matrix, int rows, int cols) {
	// Print the entire matrix
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			printf("%f ", matrix[i * cols + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void cpu_matrix_multiply(const float* A, const float* B, float* C, int M, int K, int N) {
	for (int i = 0; i < M; ++i) { // Loop over rows of A
		for (int j = 0; j < N; ++j) { // Loop over columns of B
			float sum = 0.0;
			for (int k = 0; k < K; ++k) { // Loop over columns of A / rows of B
				sum += A[i * K + k] * B[k * N + j];
			}
			C[i * N + j] = sum;
		}
	}
}

// GPU kernel for tiled matrix multiplication for non-square matrices A(MxK) * B(KxN) = C(MxN)
__global__ void revised_tiled_matrix_mult(const float* a, const float* b, float* c, int M, int K, int N) {
	// Initialize tiles
	__shared__ float tile_a[TILE_SIZE][TILE_SIZE];
	__shared__ float tile_b[TILE_SIZE][TILE_SIZE];

	// Calculate thread and block indices
	int tx = threadIdx.x; int ty = threadIdx.y;
	int bx = blockIdx.x; int by = blockIdx.y;

	// Calculate row index for matrix A and column index for matrix B
	int row = by * TILE_SIZE + ty; // Index for rows of matrix A (and C)
	int col = bx * TILE_SIZE + tx; // Index for columns of matrix B (and C)

	float p_value = 0; // Holds partial products for each value in matrix C

	// Loop over tiles of matrix A and matrix B
	for (int m = 0; m < ceil((float)K / TILE_SIZE); ++m) {
		// Load input matrix A into shared memory tile
		if (m * TILE_SIZE + tx < K && row < M) // Check if within matrix A dimensions
			tile_a[ty][tx] = a[row * K + m * TILE_SIZE + tx];
		else
			tile_a[ty][tx] = 0.0;

		// Load input matrix B into shared memory tile
		if (m * TILE_SIZE + ty < K && col < N) // Check if within matrix B dimensions
			tile_b[ty][tx] = b[(m * TILE_SIZE + ty) * N + col];
		else
			tile_b[ty][tx] = 0.0;

		__syncthreads(); // Synchronize threads to make sure tiles are loaded

		// Compute partial sum for the tile
		for (int k = 0; k < TILE_SIZE; ++k)
			p_value += tile_a[ty][k] * tile_b[k][tx];

		__syncthreads(); // Synchronize threads before loading next tiles
	}

	// Write the final result to global memory
	if (row < M && col < N)
		c[row * N + col] = p_value;
}


// Function to print the attributes of a CUDA kernel
void print_kernel_attributes() {
	cudaFuncAttributes attr;
	cudaError_t err = cudaFuncGetAttributes(&attr, revised_tiled_matrix_mult);

	if (err != cudaSuccess) {
		printf("Failed to get kernel attributes: %s\n", cudaGetErrorString(err));
		return;
	}

	int blockSize; // The launch configuration block size (number of threads per block)
	int maxActiveBlocksPerSM; // Maximum active blocks per SM

	blockSize = TILE_WIDTH * TILE_HEIGHT;

	// Calculate maximum active blocks per SM
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM, revised_tiled_matrix_mult, blockSize, 0);

	// Calculate the maximum total threads per SM
	int maxThreadsPerSM = maxActiveBlocksPerSM * blockSize;

	printf("Kernel Attributes for gpu_tiled_matrix_mult:\n");
	printf("Number of registers used by each thread: %d\n", attr.numRegs);
	printf("Shared memory per block: %zu bytes\n", attr.sharedSizeBytes);
	printf("Max active blocks per SM: %d\n", maxActiveBlocksPerSM);
	printf("Max total threads per SM: %d\n", maxThreadsPerSM);
	printf("\n");
}


void run_matrix_mult(int M_rows, int M_cols, int N_rows, int N_cols) {
	
	float* h_m = (float*)malloc(M_rows * M_cols * sizeof(float));
	float* h_n = (float*)malloc(N_rows * N_cols * sizeof(float));
	float* h_p = (float*)malloc(M_rows * N_cols * sizeof(float));
	float* d_m, * d_n, * d_p;

	// Generate random input matrices
	rand_matrix(h_m, M_rows, M_cols);
	rand_matrix(h_n, N_rows, N_cols);
	print_matrix(h_m, M_rows, M_cols);
	print_matrix(h_n, N_rows, N_cols);


	float* t_p = (float*)malloc(M_rows * M_cols * sizeof(float));
	cpu_matrix_multiply(h_m, h_n, t_p, M_rows, M_cols, N_cols);
	print_matrix(t_p, M_rows, N_cols);

	cudaMalloc(&d_m, M_rows * M_cols * sizeof(float));
	cudaMalloc(&d_n, N_rows * N_cols * sizeof(float));
	cudaMalloc(&d_p, M_rows * N_cols * sizeof(float));

	cudaMemcpy(d_m, h_m, M_rows * M_cols * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, h_n, N_rows * N_cols * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimGrid((N_cols - 1) / TILE_WIDTH + 1, (M_rows - 1) / TILE_HEIGHT + 1, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT);

	revised_tiled_matrix_mult << <dimGrid, dimBlock >> > (d_m, d_n, d_p, M_rows, M_cols, N_cols);

	cudaMemcpy(h_p, d_p, M_rows * N_cols * sizeof(float), cudaMemcpyDeviceToHost);

	print_matrix(h_p, M_rows, N_cols);


	cudaFree(d_m);
	cudaFree(d_n);
	cudaFree(d_p);
}


int main() {
	// Initialize random number generator
	srand((unsigned)time(NULL));

	// Print the CUDA kernel attributes
	print_kernel_attributes();

	run_matrix_mult(2,1,1,3);
}