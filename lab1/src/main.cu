#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CSC(call)                                               \
    do {                                                        \
        cudaError_t status = call;                              \
        if (status != cudaSuccess) {                            \
            fprintf(stderr, "[ERROR CUDA] File: '%s'; Line: %i; Message: %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(status));   \
            exit(1);                                            \
        }                                                       \
    } while (0)

#define ALLOCATE_VECTOR_CPU(host_arr, n)                        \
    double *host_arr = (double *)malloc(n * sizeof(double));    \
    if (host_arr == NULL) {                                     \
        fprintf(stderr, "[ERROR CPU] File: '%s'; Line: %i; Error in allocating mem for vector: `%s`\n", __FILE__, __LINE__, #host_arr); \
        return 1;                                               \
    }

__global__ void kernel(double *arr_1, double *arr_2, double *arr_3, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = blockDim.x * gridDim.x;
    while (idx < n) {
        arr_3[idx] = arr_1[idx] - arr_2[idx];
        idx += offset;
    }
}

int main() {

    int n, i;

    // set dim of vector
    scanf("%d", &n);

    // set mem for first vector on CPU
    ALLOCATE_VECTOR_CPU(host_arr_1, n);
    // set mem for second vector on CPU
    ALLOCATE_VECTOR_CPU(host_arr_2, n);
    // set mem for result vector on CPU
    ALLOCATE_VECTOR_CPU(host_arr_3, n);

    // fill first vector
    for (i = 0; i < n; i++) {
        scanf("%lf", &host_arr_1[i]);
    }
    // fill second vector
    for (i = 0; i < n; i++) {
        scanf("%lf", &host_arr_2[i]);
    }

    double *dev_arr_1, *dev_arr_2, *dev_arr_3;

    CSC(cudaMalloc(&dev_arr_1, sizeof(double) * n));
    CSC(cudaMalloc(&dev_arr_2, sizeof(double) * n));
    CSC(cudaMalloc(&dev_arr_3, sizeof(double) * n));

    CSC(cudaMemcpy(dev_arr_1, host_arr_1, sizeof(double) * n, cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(dev_arr_2, host_arr_2, sizeof(double) * n, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));

    CSC(cudaEventRecord(start));
    kernel<<<512, 512>>>(dev_arr_1, dev_arr_2, dev_arr_3, n);
    CSC(cudaEventRecord(stop));
    CSC(cudaEventSynchronize(stop));
    CSC(cudaGetLastError());

    float t;
    CSC(cudaEventElapsedTime(&t, start, stop));
    CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(stop));

    // printf("CUDA execution time: %f ms\n", t);

    CSC(cudaMemcpy(host_arr_3, dev_arr_3, sizeof(double) * n, cudaMemcpyDeviceToHost));

    for (i = 0; i < n; i++) {
        printf("%.10e ", host_arr_3[i]);
    }
    // printf("\n");

    CSC(cudaFree(dev_arr_1));
    CSC(cudaFree(dev_arr_2));
    CSC(cudaFree(dev_arr_3));
    free(host_arr_1);
    free(host_arr_2);
    free(host_arr_3);

    return 0;
}
