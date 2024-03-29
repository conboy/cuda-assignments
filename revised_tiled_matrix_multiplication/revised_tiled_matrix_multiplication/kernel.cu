// Author: Conrad Fernandez
// Student Number: 20219637

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <ctime>

#define TILE_WIDTH 16


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
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			float sum = 0.0;
			for (int k = 0; k < K; ++k) {
				sum += A[i * K + k] * B[k * N + j];
			}
			C[i * N + j] = sum;
		}
	}
}

bool verify_results(float* a, float* b, float* c, int a_rows, int a_cols, int b_cols) {
	// Verify matrix multiplication results (true == correct, false == incorrect)
	for (int i = 0; i < a_rows; i++) {
		for (int j = 0; j < b_cols; j++) {
			float tmp = 0.0f;
			for (int k = 0; k < a_cols; k++) {
				tmp += a[i * a_cols + k] * b[k * b_cols + j];
			}
			// Check against the CPU result
			if (tmp != c[i * b_cols + j]) return false;
		}
	}
	return true;
}

// GPU kernel for tiled matrix multiplication for non-square matrices A(MxK) * B(KxN) = C(MxN)
__global__ void revised_tiled_matrix_mult(const float* a, const float* b, float* c, int M, int K, int N) {
	// Initialize tiles
	__shared__ float tile_a[TILE_WIDTH][TILE_WIDTH];
	__shared__ float tile_b[TILE_WIDTH][TILE_WIDTH];

	// Calculate thread and block indices
	int tx = threadIdx.x; int ty = threadIdx.y;
	int bx = blockIdx.x; int by = blockIdx.y;

	// Calculate row index for matrix A and column index for matrix B
	int row = by * TILE_WIDTH + ty; // Index for rows of matrix A (and C)
	int col = bx * TILE_WIDTH + tx; // Index for columns of matrix B (and C)

	float p_value = 0; // Holds partial products for each value in matrix C

	// Loop over tiles of matrix A and matrix B
	for (int m = 0; m < ceilf((float)K / (float)TILE_WIDTH); ++m) {
		// Load input matrix A into shared memory tile
		if (row < M && (m * TILE_WIDTH + tx) < K) // Check if within matrix A dimensions
			tile_a[ty][tx] = a[row * K + (m * TILE_WIDTH + tx)];
		else
			tile_a[ty][tx] = 0.0;

		// Load input matrix B into shared memory tile
		if (col < N && (m * TILE_WIDTH + ty) < K) // Check if within matrix B dimensions
			tile_b[ty][tx] = b[(m * TILE_WIDTH + ty) * N + col];
		else
			tile_b[ty][tx] = 0.0;

		__syncthreads(); // Synchronize threads to make sure tiles are loaded

		// Compute partial sum for the tile
		for (int k = 0; k < TILE_WIDTH; ++k)
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

	blockSize = TILE_WIDTH * TILE_WIDTH;

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
	printf("Matrix Sizes: %d x %d and %d x %d\n", M_rows, M_cols, N_rows, N_cols);

	float* h_m = (float*)malloc(M_rows * M_cols * sizeof(float));
	float* h_n = (float*)malloc(N_rows * N_cols * sizeof(float));
	float* h_p = (float*)malloc(M_rows * N_cols * sizeof(float));
	float* d_m, * d_n, * d_p;

	// Generate random input matrices
	rand_matrix(h_m, M_rows, M_cols);
	rand_matrix(h_n, N_rows, N_cols);

	// Start CPU timing
	clock_t start_cpu = clock();
	cpu_matrix_multiply(h_m, h_n, h_p, M_rows, M_cols, N_cols);
	clock_t end_cpu = clock();
	double cpu_time_used = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;
	printf("CPU time: %f seconds\n", cpu_time_used);

	cudaMalloc(&d_m, M_rows * M_cols * sizeof(float));
	cudaMalloc(&d_n, N_rows * N_cols * sizeof(float));
	cudaMalloc(&d_p, M_rows * N_cols * sizeof(float));

	cudaMemcpy(d_m, h_m, M_rows * M_cols * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, h_n, N_rows * N_cols * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
	dim3 dimGrid((N_cols - 1) / TILE_WIDTH + 1, (M_rows - 1) / TILE_WIDTH + 1);

	// Prepare for GPU timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);

	revised_tiled_matrix_mult << <dimGrid, dimBlock >> > (d_m, d_n, d_p, M_rows, M_cols, N_cols);

	// End GPU timing
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("GPU time: %f seconds\n", milliseconds / 1000.0);

	cudaMemcpy(h_p, d_p, M_rows * N_cols * sizeof(float), cudaMemcpyDeviceToHost);

	// Assert test to see if CPU and GPU agree on result matrix
	if (verify_results(h_m, h_n, h_p, M_rows, M_cols, N_cols)) {
		printf("TEST PASSED\n");
	}
	else {
		printf("TEST FAILED\n");
	}

	cudaFree(d_m);
	cudaFree(d_n);
	cudaFree(d_p);
}


int main() {
	// Initialize random number generator
	srand((unsigned)time(NULL));

	// Print the CUDA kernel attributes
	print_kernel_attributes();

	run_matrix_mult(400,450,450,500);
	run_matrix_mult(1200, 1350, 1350, 1150);
}