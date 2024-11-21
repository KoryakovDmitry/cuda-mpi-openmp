#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

#define CSC(call)                                               \
    do {                                                        \
        cudaError_t status = call;                              \
        if (status != cudaSuccess) {                            \
            fprintf(stderr, "[ERROR CUDA] File: '%s'; Line: %i; Message: %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(status));   \
            exit(1);                                            \
        }                                                       \
    } while (0)

#define MEASURE_KERNEL_TIME(kernel_call, time_var)              \
    do {                                                        \
        cudaEvent_t _start, _stop;                              \
        CSC(cudaEventCreate(&_start));                          \
        CSC(cudaEventCreate(&_stop));                           \
        CSC(cudaEventRecord(_start));                           \
        kernel_call;                                            \
        CSC(cudaEventRecord(_stop));                            \
        CSC(cudaEventSynchronize(_stop));                       \
        float _elapsed_time;                                    \
        CSC(cudaEventElapsedTime(&_elapsed_time, _start, _stop)); \
        time_var += _elapsed_time;                              \
        CSC(cudaEventDestroy(_start));                          \
        CSC(cudaEventDestroy(_stop));                           \
    } while (0)

#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_GRID 256

#define MAX_CLASSES 32

// Device constants
__constant__ double3 const_avg[MAX_CLASSES];
__constant__ double const_inv_covariance[MAX_CLASSES][3][3];

// Kernel for classifying pixels
__global__ void classify_kernel(uchar4 *d_image, int w, int h, int nc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = w * h;

    if (tid < total_pixels) {
        uchar4 pixel = d_image[tid];
        double3 p = make_double3(pixel.x, pixel.y, pixel.z);

        double min_dist = DBL_MAX;
        int class_id = -1;

        for (int j = 0; j < nc; j++) {
            double3 diff = make_double3(p.x - const_avg[j].x, p.y - const_avg[j].y, p.z - const_avg[j].z);

            // Mahalanobis distance
            double dist = diff.x * (const_inv_covariance[j][0][0] * diff.x +
                                    const_inv_covariance[j][0][1] * diff.y +
                                    const_inv_covariance[j][0][2] * diff.z) +
                          diff.y * (const_inv_covariance[j][1][0] * diff.x +
                                    const_inv_covariance[j][1][1] * diff.y +
                                    const_inv_covariance[j][1][2] * diff.z) +
                          diff.z * (const_inv_covariance[j][2][0] * diff.x +
                                    const_inv_covariance[j][2][1] * diff.y +
                                    const_inv_covariance[j][2][2] * diff.z);

            if (dist < min_dist) {
                min_dist = dist;
                class_id = j;
            }
        }

        d_image[tid].w = class_id; // Write class ID to alpha channel
    }
}

// Host function
int main() {
    // Input and output file paths
    char input_file[4096], output_file[4096];

    // Reading input file paths with buffer size limits
    if (scanf("%4095s", input_file) != 1) {
        fprintf(stderr, "Error reading input filepath.\n");
        return 1;
    }

    // Reading output file paths with buffer size limits
    if (scanf("%4095s", output_file) != 1) {
        fprintf(stderr, "Error reading output filepath.\n");
        return 1;
    }

    // Read image dimensions
    int w, h;
    FILE *fin = fopen(input_file, "rb");
    fread(&w, sizeof(int), 1, fin);
    fread(&h, sizeof(int), 1, fin);

    // Allocate host and device memory for the image
    int total_pixels = w * h;
    uchar4 *h_image = (uchar4 *)malloc(total_pixels * sizeof(uchar4));
    uchar4 *d_image;
    CSC(cudaMalloc(&d_image, total_pixels * sizeof(uchar4)));

    // Read image data
    fread(h_image, sizeof(uchar4), total_pixels, fin);
    fclose(fin);
    CSC(cudaMemcpy(d_image, h_image, total_pixels * sizeof(uchar4), cudaMemcpyHostToDevice));

    // Number of classes
    int nc;
    scanf("%d", &nc);

    // Host memory for averages and covariances
    double3 avg[MAX_CLASSES];
    double covariance[MAX_CLASSES][3][3];

    // Read class data and compute averages and covariances
    for (int j = 0; j < nc; j++) {
        int npj;
        scanf("%d", &npj);

        double3 sum = make_double3(0, 0, 0);
        double3 *pixels = (double3 *)malloc(npj * sizeof(double3));

        for (int i = 0; i < npj; i++) {
            int x, y;
            scanf("%d %d", &x, &y);
            uchar4 pixel = h_image[y * w + x];
            double3 p = make_double3(pixel.x, pixel.y, pixel.z);
            sum.x += p.x;
            sum.y += p.y;
            sum.z += p.z;
            pixels[i] = p;
        }

        avg[j] = make_double3(sum.x / npj, sum.y / npj, sum.z / npj);

        double cov[3][3] = {0};
        for (int i = 0; i < npj; i++) {
            double3 diff = make_double3(pixels[i].x - avg[j].x, pixels[i].y - avg[j].y, pixels[i].z - avg[j].z);
            cov[0][0] += diff.x * diff.x;
            cov[0][1] += diff.x * diff.y;
            cov[0][2] += diff.x * diff.z;
            cov[1][0] += diff.y * diff.x;
            cov[1][1] += diff.y * diff.y;
            cov[1][2] += diff.y * diff.z;
            cov[2][0] += diff.z * diff.x;
            cov[2][1] += diff.z * diff.y;
            cov[2][2] += diff.z * diff.z;
        }

        for (int i = 0; i < 3; i++)
            for (int k = 0; k < 3; k++)
                covariance[j][i][k] = cov[i][k] / (npj - 1);

        free(pixels);
    }

    // Copy averages and inverse covariances to device
    CSC(cudaMemcpyToSymbol(const_avg, avg, nc * sizeof(double3)));
    CSC(cudaMemcpyToSymbol(const_inv_covariance, covariance, nc * sizeof(double[3][3])));

    // Launch classification kernel
    float total_kernel_time = 0.0f;
    MEASURE_KERNEL_TIME(classify_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_image, w, h, nc));
    CSC(cudaDeviceSynchronize());

    // Copy results back to host and write output
    CSC(cudaMemcpy(h_image, d_image, total_pixels * sizeof(uchar4), cudaMemcpyDeviceToHost));
    FILE *fout = fopen(output_file, "wb");
    fwrite(&w, sizeof(int), 1, fout);
    fwrite(&h, sizeof(int), 1, fout);
    fwrite(h_image, sizeof(uchar4), total_pixels, fout);
    fclose(fout);

    // Free memory
    free(h_image);
    CSC(cudaFree(d_image));
    // printf("CUDA execution time: <%f ms>\n", total_kernel_time);
    return 0;
}
