#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>
#include <string.h>

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

__global__ void classify_kernel(uchar4 *d_image, int w, int h, int nc) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = w * h;

    while (tid < total_pixels) {
        uchar4 pixel = d_image[tid];
        double min_distance = DBL_MAX;
        int best_class = -1;

        for (int c = 0; c < nc; ++c) {
            double diff[3] = {
                pixel.x - const_avg[c].x,
                pixel.y - const_avg[c].y,
                pixel.z - const_avg[c].z
            };

            double temp[3] = {0, 0, 0};
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    temp[i] += diff[j] * const_inv_covariance[c][i][j];
                }
            }

            double distance = 0;
            for (int i = 0; i < 3; ++i) {
                distance += temp[i] * diff[i];
            }

            if (distance < min_distance) {
                min_distance = distance;
                best_class = c;
            }
        }

        d_image[tid].w = best_class; // Store class in alpha channel
        tid += blockDim.x * gridDim.x;
    }
}

int main() {
    int w, h, nc;
    char inputFilepath[4096], outputFilepath[4096];

    // Reading input and output file paths
    if (scanf("%4095s", inputFilepath) != 1 || scanf("%4095s", outputFilepath) != 1) {
        fprintf(stderr, "Error reading file paths.\n");
        return 1;
    }

    // Open input file
    FILE *input_file = fopen(inputFilepath, "rb");
    if (!input_file) {
        fprintf(stderr, "Error opening input file.\n");
        return 1;
    }

    fread(&w, sizeof(int), 1, input_file);
    fread(&h, sizeof(int), 1, input_file);

    uchar4 *h_image = (uchar4 *)malloc(w * h * sizeof(uchar4));
    fread(h_image, sizeof(uchar4), w * h, input_file);
    fclose(input_file);

    double3 h_avg[MAX_CLASSES];
    double h_covariance[MAX_CLASSES][3][3];
    double h_inv_covariance[MAX_CLASSES][3][3];

    // Reading the number of classes
    if (scanf("%d", &nc) != 1) {
        fprintf(stderr, "Error reading number of classes.\n");
        return 1;
    }

    // Reading class information
    for (int c = 0; c < nc; ++c) {
        int npixels;
        if (scanf("%d", &npixels) != 1 || npixels <= 0) {
            fprintf(stderr, "Invalid number of pixels for class %d.\n", c);
            free(h_image);
            return 1;
        }

        double sum_x = 0, sum_y = 0, sum_z = 0;
        for (int p = 0; p < npixels; ++p) {
            int x, y;
            scanf("%d %d", &x, &y);
            uchar4 pixel = h_image[y * w + x];
            sum_x += pixel.x;
            sum_y += pixel.y;
            sum_z += pixel.z;
        }

        h_avg[c].x = sum_x / npixels;
        h_avg[c].y = sum_y / npixels;
        h_avg[c].z = sum_z / npixels;

        // Reset covariance matrix
        memset(h_covariance[c], 0, 9 * sizeof(double));
    }

    // Covariance and inversion (left as placeholder, implement accordingly)
    // Copy data to device constant memory
    CSC(cudaMemcpyToSymbol(const_avg, h_avg, sizeof(double3) * nc));
    CSC(cudaMemcpyToSymbol(const_inv_covariance, h_inv_covariance, sizeof(double) * 9 * nc));

    uchar4 *d_image;
    CSC(cudaMalloc(&d_image, w * h * sizeof(uchar4)));
    CSC(cudaMemcpy(d_image, h_image, w * h * sizeof(uchar4), cudaMemcpyHostToDevice));

    float total_kernel_time = 0.0f;
    MEASURE_KERNEL_TIME(
        (classify_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_image, w, h, nc)),
        total_kernel_time);

    CSC(cudaMemcpy(h_image, d_image, w * h * sizeof(uchar4), cudaMemcpyDeviceToHost));
    CSC(cudaFree(d_image));

    FILE *output_file = fopen(outputFilepath, "wb");
    if (!output_file) {
        fprintf(stderr, "Error opening output file.\n");
        free(h_image);
        return 1;
    }

    fwrite(&w, sizeof(int), 1, output_file);
    fwrite(&h, sizeof(int), 1, output_file);
    fwrite(h_image, sizeof(uchar4), w * h, output_file);
    fclose(output_file);

    free(h_image);
    return 0;
}
