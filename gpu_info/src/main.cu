#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    int device = 0;
    cudaGetDeviceProperties(&prop, device);

    printf("Compute capability : %d.%d\n", prop.major, prop.minor);
    printf("Name : %s\n", prop.name);
    printf("Total Global Memory : %zu\n", prop.totalGlobalMem);
    printf("Shared memory per block : %zu\n", prop.sharedMemPerBlock);
    printf("Max threads per block : (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max block : (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Total constant memory : %zu\n", prop.totalConstMem);
    printf("Multiprocessors count : %d\n", prop.multiProcessorCount);

    return 0;
}
