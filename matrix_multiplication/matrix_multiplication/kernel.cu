#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 100	// Matrix size

void rand_matrix(float *matrix, int n) {
	// Fill a N x N matrix with random floats
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			float randomNumber = 1 + (float)rand() / ((float)RAND_MAX / (255 - 1));
			matrix[i * n + j] = randomNumber;
		}
	}
}

void print_matrix(float *matrix, int n) {
	// Print the entire matrix
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%f ", matrix[i * n + j]);
		}
		printf("\n");
	}
}

void cpu_matrix_mult(float *a, float *b, float *c, int n) {
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
			if (tmp != c[i * N + j]) return false;
		}
	}
	return true;
}

// GPU kernel for matrix multiplication
__global__ void gpu_matrix_mult(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index for the element to compute
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index for the element to compute

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}


int main() {
	// Initialize random number generator
	srand((unsigned)time(NULL));

	// Initialize matrices
	float *h_m = (float *)malloc(N*N*sizeof(float));
	float *h_n = (float *)malloc(N*N*sizeof(float));
	float *h_p = (float *)malloc(N*N*sizeof(float));
	float *d_m, *d_n, *d_p;

	// Generate random input matrices
	rand_matrix(h_m, N);
	rand_matrix(h_n, N);

	// Allocate device memory
	cudaMalloc((void **)&d_m, N*N*sizeof(float));
	cudaMalloc((void **)&d_n, N*N*sizeof(float));
	cudaMalloc((void **)&d_p, N*N*sizeof(float));
	
	// Set kernel launch config
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

	// Create cuda event handles
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaDeviceSynchronize();
	float gpu_time = 0.0f;

	// Asynchronously copy memory from host to device (all to stream 0)
	cudaEventRecord(start, 0);
	cudaMemcpyAsync(d_m, h_m, N * N * sizeof(float), cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(d_n, h_n, N * N * sizeof(float), cudaMemcpyHostToDevice, 0);
	cudaEventRecord(stop, 0);

	// have CPU do some work while waiting for GPU to finish
	unsigned long int counter = 0;
	while (cudaEventQuery(stop) == cudaErrorNotReady)
	{
		counter++; // Indicates that the CPU is running asynchronously while GPU is executing
	}
	cudaEventSynchronize(stop); // stop is updated here
	cudaEventElapsedTime(&gpu_time, start, stop); //time difference between start and stop

	// print the GPU times
	printf("\nTime spent transferring matrices to the GPU: %f\n", gpu_time);
	printf("CPU executed %d iterations while waiting for GPU to finish\n", counter);

	// Launch kernel
	gpu_matrix_mult<<<blocksPerGrid, threadsPerBlock >>>(d_m, d_n, d_p, N);

	// Copy memory from device to host
	cudaMemcpy(h_p, d_p, N*N*sizeof(float), cudaMemcpyDeviceToHost);

	// Assert test to see if CPU and GPU agree on result matrix
	if (verify_results(h_m, h_n, h_p, N)) {
		printf("TEST PASSED");
	}
	else {
		printf("TEST FAILED");
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