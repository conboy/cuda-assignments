// Author: Conrad Fernandez
// Student Number: 20219637

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE_SIZE 25

void rand_matrix(float* matrix, int n) {
	// Fill a N x N matrix with random floats
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			float randomNumber = 1 + (float)rand() / ((float)RAND_MAX / (255 - 1));
			matrix[i * n + j] = randomNumber;
		}
	}
}

void print_matrix(float* matrix, int n) {
	// Print the entire matrix
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("%f ", matrix[i * n + j]);
		}
		printf("\n");
	}
}

void cpu_matrix_mult(float* a, float* b, float* c, int n) {
	// Matrix multiplication on the CPU
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			c[i * n + j] = 0;
			for (int k = 0; k < n; k++) {
				c[i * n + j] += a[i * n + k] * b[k * n + j]; // Perform the dot product of row i of A and column j of B
			}
		}
	}
}

bool verify_results(float* a, float* b, float* c, int n) {
	// Verify matrix multiplication results (0 == correct, 1 == incorrect)
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			float tmp = 0;
			for (int k = 0; k < n; k++) {
				tmp += a[i * n + k] * b[k * n + j]; // Perform the dot product of row i of A and column j of B
			}
			// Check against the CPU result
			if (tmp != c[i * n + j]) return false;
		}
	}
	return true;
}

// GPU kernel for tiled matrix multiplication
__global__ void gpu_tiled_matrix_mult(const float* a, const float* b, float* c, int n) {
	// Initialize tiles
	__shared__ float tile_a[TILE_SIZE][TILE_SIZE];
	__shared__ float tile_b[TILE_SIZE][TILE_SIZE];

	// Calculate thread and block indices
	int tx = threadIdx.x; int ty = threadIdx.y;
	int bx = blockIdx.x; int by = blockIdx.y;

	// Calculate row and column indices
	int row = by * TILE_SIZE + ty;
	int col = bx * TILE_SIZE + tx;

	float p_value = 0;	// Holds partial products for each values in matrix

	// Loop over tiles
	for (int i = 0; i < ceil((float)n / TILE_SIZE); i++) {
		// Load input matrix A into tiles
		tile_a[ty][tx] = a[row * n + i * TILE_SIZE + tx];
		// Load input matrix B into tiles
		tile_b[ty][tx] = b[(i * TILE_SIZE + ty) * n + col];

		__syncthreads(); // Synchronize threads after loading tiles

		// Compute partial sum for the tile
		for (int i = 0; i < TILE_SIZE; i++)
			p_value += tile_a[ty][i] * tile_b[i][tx];

		__syncthreads(); // Synchronize threads before loading next tiles
	}

	// Write the final result to global memory
	if (row < n && col < n)
		c[row * n + col] = p_value;
}

// Function to print the attributes of a CUDA kernel
void print_kernel_attributes() {
	cudaFuncAttributes attr;
	cudaError_t err = cudaFuncGetAttributes(&attr, gpu_tiled_matrix_mult);

	if (err != cudaSuccess) {
		printf("Failed to get kernel attributes: %s\n", cudaGetErrorString(err));
		return;
	}

	int blockSize; // The launch configuration block size (number of threads per block)
	int maxActiveBlocksPerSM; // Maximum active blocks per SM

	blockSize = TILE_SIZE * TILE_SIZE;

	// Calculate maximum active blocks per SM
	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocksPerSM, gpu_tiled_matrix_mult, blockSize, 0);

	// Calculate the maximum total threads per SM
	int maxThreadsPerSM = maxActiveBlocksPerSM * blockSize;

	printf("Kernel Attributes for gpu_tiled_matrix_mult:\n");
	printf("Tile Size: %d x %d\n", TILE_SIZE, TILE_SIZE);
	printf("Number of registers used by each thread: %d\n", attr.numRegs);
	printf("Shared memory per block: %zu bytes\n", attr.sharedSizeBytes);
	printf("Max active blocks per SM: %d\n", maxActiveBlocksPerSM);
	printf("Max total threads per SM: %d\n", maxThreadsPerSM);
	printf("\n");
}


void run_matrix_mult(int n) {
	printf("%d x %d,", n, n);
	printf("%d,", TILE_SIZE);

	// Initialize matrices
	float* h_m = (float*)malloc(n * n * sizeof(float));
	float* h_n = (float*)malloc(n * n * sizeof(float));
	float* h_p = (float*)malloc(n * n * sizeof(float));
	float* d_m, * d_n, * d_p;

	// Generate random input matrices
	rand_matrix(h_m, n);
	rand_matrix(h_n, n);

	// Allocate device memory
	cudaMalloc((void**)&d_m, n * n * sizeof(float));
	cudaMalloc((void**)&d_n, n * n * sizeof(float));
	cudaMalloc((void**)&d_p, n * n * sizeof(float));

	// Set kernel launch config
	dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
	dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Create cuda event handles
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaDeviceSynchronize(); // Ensure GPU is ready

	// Copy memory from host to device
	cudaMemcpy(d_m, h_m, n * n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, h_n, n * n * sizeof(float), cudaMemcpyHostToDevice);

	// Record the start event
	cudaEventRecord(start, 0);

	// Launch the kernel
	gpu_tiled_matrix_mult << <blocksPerGrid, threadsPerBlock >> > (d_m, d_n, d_p, n);

	// Record the stop event immediately after kernel launch
	cudaEventRecord(stop, 0);

	// Wait for the kernel to complete
	cudaEventSynchronize(stop);

	// Calculate the elapsed time between events
	float gpu_time = 0.0f;
	cudaEventElapsedTime(&gpu_time, start, stop);

	// Output the time spent by the GPU executing the kernel
	printf("%f,", gpu_time);

	// Copy the result matrix back to the host (not part of timing)
	cudaMemcpy(h_p, d_p, n * n * sizeof(float), cudaMemcpyDeviceToHost);

	// Assert test to see if CPU and GPU agree on result matrix
	if (verify_results(h_m, h_n, h_p, n)) {
		printf("TEST PASSED\n");
	}
	else {
		printf("TEST FAILED\n");
	}

	// Free CUDA Events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	// Free host memory
	cudaFreeHost(h_m);
	cudaFreeHost(h_n);
	cudaFreeHost(h_p);
	// Free device memory
	cudaFree(d_m);
	cudaFree(d_n);
	cudaFree(d_p);
}


int main() {
	// Initialize random number generator
	srand((unsigned)time(NULL));

	// Print the CUDA kernel attributes
	print_kernel_attributes();

	for (int i = 0; i < 5; i++) {
		run_matrix_mult(100);
		run_matrix_mult(250);
		run_matrix_mult(500);
		run_matrix_mult(1000);
		run_matrix_mult(1500);
	}
}