#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

// Макрос для проверки ошибок CUDA
#define CSC(call) do {                                                         \
    cudaError_t status = call;                                                 \
    if (status != cudaSuccess) {                                               \
        fprintf(stderr, "[ERROR CUDA] File: '%s'; Line: %i; Message: %s.\n",   \
                __FILE__, __LINE__, cudaGetErrorString(status));               \
        exit(1);                                                               \
    }                                                                          \
} while (0)

#define MEASURE_KERNEL_TIME(kernel_call, time_var) do {                        \
    cudaEvent_t _start, _stop;                                                 \
    CSC(cudaEventCreate(&_start));                                             \
    CSC(cudaEventCreate(&_stop));                                              \
    CSC(cudaEventRecord(_start));                                              \
    kernel_call;                                                               \
    CSC(cudaEventRecord(_stop));                                               \
    CSC(cudaEventSynchronize(_stop));                                          \
    float _elapsed_time;                                                       \
    CSC(cudaEventElapsedTime(&_elapsed_time, _start, _stop));                  \
    time_var += _elapsed_time;                                                 \
    CSC(cudaEventDestroy(_start));                                             \
    CSC(cudaEventDestroy(_stop));                                              \
} while (0)

#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_GRID 256

#define MAX_CLASSES 32

__constant__ double3 const_avg[MAX_CLASSES];
__constant__ double const_inv_cov[MAX_CLASSES][3][3];

__global__ void classify_kernel(uchar4 *d_image, int w, int h, int nc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = w * h;

    while (idx < total_pixels) {
        uchar4 pixel = d_image[idx];
        double min_dist = DBL_MAX;
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
                    temp[i] += diff[j] * const_inv_cov[c][j][i];
                }
            }

            double dist = 0;
            for (int i = 0; i < 3; ++i) {
                dist += temp[i] * diff[i];
            }

            if (dist < min_dist) {
                min_dist = dist;
                best_class = c;
            }
        }
        d_image[idx].w = best_class;
        idx += blockDim.x * gridDim.x;
    }
}

int main() {
    int w, h, nc;
    char input_file[4096], output_file[4096];
    scanf("%4095s", input_file);
    scanf("%4095s", output_file);

    FILE *fin = fopen(input_file, "rb");
    fread(&w, sizeof(int), 1, fin);
    fread(&h, sizeof(int), 1, fin);

    int total_pixels = w * h;
    uchar4 *h_image = (uchar4 *)malloc(total_pixels * sizeof(uchar4));
    fread(h_image, sizeof(uchar4), total_pixels, fin);
    fclose(fin);

    uchar4 *d_image;
    CSC(cudaMalloc(&d_image, total_pixels * sizeof(uchar4)));
    CSC(cudaMemcpy(d_image, h_image, total_pixels * sizeof(uchar4), cudaMemcpyHostToDevice));

    scanf("%d", &nc);
    double3 avg[MAX_CLASSES];
    double cov[MAX_CLASSES][3][3] = {0};
    double inv_cov[MAX_CLASSES][3][3] = {0};

    for (int c = 0; c < nc; ++c) {
        int np;
        scanf("%d", &np);

        double sum[3] = {0, 0, 0};
        int *coords = (int *)malloc(np * 2 * sizeof(int));
        for (int i = 0; i < np; ++i) {
            scanf("%d %d", &coords[i * 2], &coords[i * 2 + 1]);
            int idx = coords[i * 2 + 1] * w + coords[i * 2];
            uchar4 px = h_image[idx];
            sum[0] += px.x;
            sum[1] += px.y;
            sum[2] += px.z;
        }

        avg[c] = make_double3(sum[0] / np, sum[1] / np, sum[2] / np);

        for (int i = 0; i < np; ++i) {
            int idx = coords[i * 2 + 1] * w + coords[i * 2];
            uchar4 px = h_image[idx];
            double diff[3] = {
                px.x - avg[c].x,
                px.y - avg[c].y,
                px.z - avg[c].z
            };

            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    cov[c][a][b] += diff[a] * diff[b];
                }
            }
        }

        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                cov[c][a][b] /= (np - 1);
            }
        }

        double det = cov[c][0][0] * (cov[c][1][1] * cov[c][2][2] - cov[c][2][1] * cov[c][1][2])
                   - cov[c][0][1] * (cov[c][1][0] * cov[c][2][2] - cov[c][1][2] * cov[c][2][0])
                   + cov[c][0][2] * (cov[c][1][0] * cov[c][2][1] - cov[c][1][1] * cov[c][2][0]);

        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                inv_cov[c][a][b] = (cov[c][(b + 1) % 3][(a + 1) % 3] * cov[c][(b + 2) % 3][(a + 2) % 3]
                                   - cov[c][(b + 1) % 3][(a + 2) % 3] * cov[c][(b + 2) % 3][(a + 1) % 3]) / det;
            }
        }
        free(coords);
    }

    CSC(cudaMemcpyToSymbol(const_avg, avg, nc * sizeof(double3)));
    CSC(cudaMemcpyToSymbol(const_inv_cov, inv_cov, nc * sizeof(double[3][3])));

    float total_time = 0;
    MEASURE_KERNEL_TIME(classify_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_image, w, h, nc), total_time);

    CSC(cudaMemcpy(h_image, d_image, total_pixels * sizeof(uchar4), cudaMemcpyDeviceToHost));
    CSC(cudaFree(d_image));

    FILE *fout = fopen(output_file, "wb");
    fwrite(&w, sizeof(int), 1, fout);
    fwrite(&h, sizeof(int), 1, fout);
    fwrite(h_image, sizeof(uchar4), total_pixels, fout);
    fclose(fout);

    free(h_image);
    //printf("CUDA execution time: %f ms\n", total_time);
    return 0;
}
