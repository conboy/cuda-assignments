// Author: Conrad Fernandez
// Student Number: 20219637

#include <cuda_runtime.h>
#include <stdio.h>

int GetCoresPerSM(int major, int minor) {
    // Defines for GPU Architecture types (using NVIDIA's definitions)
    if (major == 1) { // Tesla
        return 8;
    }
    else if (major == 2) {
        if (minor == 0) return 32; // Fermi Generation
        else return 48;
    }
    else if (major == 3) { // Kepler
        return 192;
    }
    else if (major == 5) { // Maxwell
        return 128;
    }
    else if (major == 6) { // Pascal
        if (minor == 0 || minor == 2) return 64;
        else if (minor == 1) return 128;
    }
    else if (major == 7) { // Volta and Turing
        if (minor == 0) return 64;
        else if (minor == 5) return 64;
    }
    else if (major == 8) { // Ampere
        if (minor == 0 || minor == 6) return 64;
        else if (minor == 6) return 128;
    }
    return -1; // Default to -1 cores per SM for unknown architectures
}

int main() {
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    printf("CUDA devices available: %d\n\n", nDevices);

    for (int i = 0; i < nDevices; i++) {
        // Initialize and get the device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        // Calculate core count
        int nCores = GetCoresPerSM(prop.major, prop.minor) * prop.multiProcessorCount;

        printf("Device Number: %d\n", i);
        printf("    Device Name: %s\n", prop.name);
        printf("    Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("    Clock Rate: %d kHz\n", prop.clockRate);
        printf("    Number of streaming multiprocessors: %d\n", prop.multiProcessorCount);
        printf("    Number of cores: %d\n", nCores);
        printf("    Warp size: %d\n", prop.warpSize);
        printf("    Global Memory: %.2f GB\n", (float)prop.totalGlobalMem / (1024 * 1024 * 1024));
        printf("    Constant Memory: %.2f KB\n", (float)prop.totalConstMem / 1024);
        printf("    Shared Memory per Block: %.2f KB\n", (float)prop.sharedMemPerBlock / 1024);
        printf("    Registers per Block: %d\n", prop.regsPerBlock);
        printf("    Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("    Max Thread Dimensions: (%d, %d, %d)\n",
            prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("    Max Grid Dimensions: (%d, %d, %d)\n",
            prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    return 0;
}
