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

// Macro to measure kernel execution time
#define MEASURE_KERNEL_TIME(kernel_call, time_var)          \
    do {                                                    \
        cudaEvent_t _start, _stop;                          \
        CSC(cudaEventCreate(&_start));                      \
        CSC(cudaEventCreate(&_stop));                       \
        CSC(cudaEventRecord(_start));                       \
        kernel_call;                                        \
        CSC(cudaEventRecord(_stop));                        \
        CSC(cudaEventSynchronize(_stop));                   \
        float _elapsed_time;                                \
        CSC(cudaEventElapsedTime(&_elapsed_time, _start, _stop)); \
        time_var += _elapsed_time;                          \
        CSC(cudaEventDestroy(_start));                      \
        CSC(cudaEventDestroy(_stop));                       \
    } while (0)

#define MAX_CLASSES 32

// Constants for blocks, threads, and grids
const int THREADS_PER_BLOCK_X = 32;
const int THREADS_PER_BLOCK_Y = 32;
const int BLOCKS_PER_GRID_X = 16;
const int BLOCKS_PER_GRID_Y = 16;

// Constant memory for averages and inverse covariance matrices
__constant__ double const_avg_r[MAX_CLASSES];
__constant__ double const_avg_g[MAX_CLASSES];
__constant__ double const_avg_b[MAX_CLASSES];
__constant__ double const_inv_covariance_matrices[MAX_CLASSES][3][3];

// Device function to invert a 3x3 matrix (double precision)
__device__ void invert_3x3_matrix_double(const double *a, double *inv_a) {
    double det = a[0]*(a[4]*a[8] - a[5]*a[7]) - a[1]*(a[3]*a[8] - a[5]*a[6]) + a[2]*(a[3]*a[7] - a[4]*a[6]);

    if (fabs(det) < 1e-6) {
        // Matrix is singular, set inverse to zero matrix
        for (int i = 0; i < 9; i++)
            inv_a[i] = 0.0;
        return;
    }

    double inv_det = 1.0 / det;

    inv_a[0] = (a[4]*a[8] - a[5]*a[7]) * inv_det;
    inv_a[1] = (a[2]*a[7] - a[1]*a[8]) * inv_det;
    inv_a[2] = (a[1]*a[5] - a[2]*a[4]) * inv_det;

    inv_a[3] = (a[5]*a[6] - a[3]*a[8]) * inv_det;
    inv_a[4] = (a[0]*a[8] - a[2]*a[6]) * inv_det;
    inv_a[5] = (a[2]*a[3] - a[0]*a[5]) * inv_det;

    inv_a[6] = (a[3]*a[7] - a[4]*a[6]) * inv_det;
    inv_a[7] = (a[1]*a[6] - a[0]*a[7]) * inv_det;
    inv_a[8] = (a[0]*a[4] - a[1]*a[3]) * inv_det;
}

// Kernel to read sample pixel values
__global__ void read_sample_pixels(uchar4 *d_image, int w, int h, int total_npj, int *d_coordinates_flat, float3 *d_sample_pixels) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= total_npj)
        return;

    int x = d_coordinates_flat[tid * 2];
    int y = d_coordinates_flat[tid * 2 + 1];

    // Check if x and y are within image bounds
    if (x < 0 || x >= w || y < 0 || y >= h) {
        d_sample_pixels[tid] = make_float3(0.0f, 0.0f, 0.0f);
    } else {
        uchar4 p = d_image[y * w + x];
        d_sample_pixels[tid] = make_float3((float)p.x, (float)p.y, (float)p.z);
    }
}

// Kernel to compute sums for means
__global__ void compute_sums(int total_npj, int *d_class_ids, float3 *d_sample_pixels, double *d_sums_r, double *d_sums_g, double *d_sums_b) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= total_npj)
        return;

    int class_id = d_class_ids[tid];

    float3 p = d_sample_pixels[tid];

    atomicAdd(&d_sums_r[class_id], (double)p.x);
    atomicAdd(&d_sums_g[class_id], (double)p.y);
    atomicAdd(&d_sums_b[class_id], (double)p.z);
}

// Kernel to compute averages
__global__ void compute_averages(int nc, double *d_sums_r, double *d_sums_g, double *d_sums_b, int *d_npjs, double *d_avg_r, double *d_avg_g, double *d_avg_b) {
    int c = threadIdx.x + blockIdx.x * blockDim.x;

    if (c >= nc)
        return;

    int npj = d_npjs[c];

    if (npj > 0) {
        d_avg_r[c] = d_sums_r[c] / npj;
        d_avg_g[c] = d_sums_g[c] / npj;
        d_avg_b[c] = d_sums_b[c] / npj;
    } else {
        d_avg_r[c] = 0.0;
        d_avg_g[c] = 0.0;
        d_avg_b[c] = 0.0;
    }
}

// Kernel to compute covariance matrices
__global__ void compute_covariances(int total_npj, int *d_class_ids, float3 *d_sample_pixels, double *d_avg_r, double *d_avg_g, double *d_avg_b, double *d_covariance_matrices) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= total_npj)
        return;

    int class_id = d_class_ids[tid];

    float3 p = d_sample_pixels[tid];

    double3 avg;
    avg.x = d_avg_r[class_id];
    avg.y = d_avg_g[class_id];
    avg.z = d_avg_b[class_id];

    double3 diff;
    diff.x = (double)p.x - avg.x;
    diff.y = (double)p.y - avg.y;
    diff.z = (double)p.z - avg.z;

    // Indices of covariance matrix elements:
    // [0 1 2]
    // [3 4 5]
    // [6 7 8]

    double *cov = &d_covariance_matrices[class_id * 9];

    atomicAdd(&cov[0], diff.x * diff.x); // Cxx
    atomicAdd(&cov[1], diff.x * diff.y); // Cxy
    atomicAdd(&cov[2], diff.x * diff.z); // Cxz
    atomicAdd(&cov[4], diff.y * diff.y); // Cyy
    atomicAdd(&cov[5], diff.y * diff.z); // Cyz
    atomicAdd(&cov[8], diff.z * diff.z); // Czz
}

// Kernel to finalize covariance matrices
__global__ void finalize_covariances(int nc, double *d_covariance_matrices, int *d_npjs) {
    int c = threadIdx.x + blockIdx.x * blockDim.x;

    if (c >= nc)
        return;

    int npj = d_npjs[c];

    if (npj > 1) {
        double inv_np1 = 1.0 / (npj - 1);

        double *cov = &d_covariance_matrices[c * 9];

        // Finalize covariance matrix
        cov[0] *= inv_np1; // Cxx
        cov[1] *= inv_np1; // Cxy
        cov[2] *= inv_np1; // Cxz
        cov[4] *= inv_np1; // Cyy
        cov[5] *= inv_np1; // Cyz
        cov[8] *= inv_np1; // Czz

        // Set symmetric elements
        cov[3] = cov[1]; // Cyx = Cxy
        cov[6] = cov[2]; // Czx = Cxz
        cov[7] = cov[5]; // Czy = Cyz
    } else {
        // If npj <= 1, set covariance matrix to identity
        double *cov = &d_covariance_matrices[c * 9];

        for (int i = 0; i < 9; i++) {
            cov[i] = 0.0;
        }
    }
}

// Kernel to invert covariance matrices
__global__ void invert_covariances(int nc, double *d_covariance_matrices, double *d_inverse_covariance_matrices) {
    int c = threadIdx.x + blockIdx.x * blockDim.x;

    if (c >= nc)
        return;

    double *cov = &d_covariance_matrices[c * 9];
    double *inv_cov = &d_inverse_covariance_matrices[c * 9];

    invert_3x3_matrix_double(cov, inv_cov);
}

// Main kernel to compute Mahalanobis distances and assign class labels
__global__ void classify_kernel(uchar4 *d_image, int w, int h, int nc) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int y = idy; y < h; y += offsety) {
        for (int x = idx; x < w; x += offsetx) {
            uchar4 p_uchar = d_image[y * w + x];
            double3 p;
            p.x = (double)p_uchar.x;
            p.y = (double)p_uchar.y;
            p.z = (double)p_uchar.z;

            double min_m = DBL_MAX;
            int label_class_idx_int = -1;

            for (int c = 0; c < nc; c++) {
                double3 avg_j;
                avg_j.x = const_avg_r[c];
                avg_j.y = const_avg_g[c];
                avg_j.z = const_avg_b[c];

                double3 diff;
                diff.x = p.x - avg_j.x;
                diff.y = p.y - avg_j.y;
                diff.z = p.z - avg_j.z;

                const double (*inv_cov)[3] = const_inv_covariance_matrices[c];

                double3 temp;
                temp.x = inv_cov[0][0] * diff.x + inv_cov[0][1] * diff.y + inv_cov[0][2] * diff.z;
                temp.y = inv_cov[1][0] * diff.x + inv_cov[1][1] * diff.y + inv_cov[1][2] * diff.z;
                temp.z = inv_cov[2][0] * diff.x + inv_cov[2][1] * diff.y + inv_cov[2][2] * diff.z;

                double m = diff.x * temp.x + diff.y * temp.y + diff.z * temp.z;

                if (m < min_m) {
                    min_m = m;
                    label_class_idx_int = c;
                } else if (m == min_m && c < label_class_idx_int) {
                    label_class_idx_int = c;
                }
            }

            // Convert to unsigned char
            unsigned char label_class = static_cast<unsigned char>(label_class_idx_int);

            // Set the output pixel alpha channel to the class label
            d_image[y * w + x].w = label_class;
        }
    }
}

int main() {
    int w, h;
    int nc; // Number of classes

    char inputFilepath[4096], outputFilepath[4096];
    scanf("%4095s", inputFilepath);
    scanf("%4095s", outputFilepath);

    // Reading the number of classes
    scanf("%d", &nc);

    // Memory allocation for the number of pixels in each class
    int *npjs = (int *)malloc(nc * sizeof(int));
    if (npjs == NULL) {
        fprintf(stderr, "Memory allocation error!\n");
        return 1;
    }

    // Arrays to store the coordinates of pixels (2D dynamic array)
    int **coordinates = (int **)malloc(nc * sizeof(int *));
    if (coordinates == NULL) {
        fprintf(stderr, "Memory allocation error!\n");
        free(npjs);
        return 1;
    }

    // Reading data for each class
    for (int c = 0; c < nc; c++) {
        // Reading the number of pixels
        scanf("%d", &npjs[c]);

        // Allocating memory to store coordinates (npjs[c] pairs of numbers)
        coordinates[c] = (int *)malloc(npjs[c] * 2 * sizeof(int));
        if (coordinates[c] == NULL) {
            fprintf(stderr, "Memory allocation error!\n");
            for (int i = 0; i < c; i++) {
                free(coordinates[i]);
            }
            free(coordinates);
            free(npjs);
            return 1;
        }

        // Reading the coordinates of pixels
        for (int p = 0; p < npjs[c]; p++) {
            scanf("%d %d", &coordinates[c][p * 2], &coordinates[c][p * 2 + 1]);
        }
    }

    FILE *fp = fopen(inputFilepath, "rb");
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    // Allocate device memory for image data
    uchar4 *d_image;
    CSC(cudaMalloc(&d_image, w * h * sizeof(uchar4)));
    CSC(cudaMemcpy(d_image, data, w * h * sizeof(uchar4), cudaMemcpyHostToDevice));

    // Prepare data for processing
    int total_npj = 0;
    int *offsets = (int *)malloc((nc + 1) * sizeof(int)); // offsets[0..nc]
    offsets[0] = 0;
    for (int c = 0; c < nc; c++) {
        offsets[c + 1] = offsets[c] + npjs[c];
    }
    total_npj = offsets[nc]; // total number of sample pixels

    // Flatten coordinates
    int *coordinates_flat = (int *)malloc(total_npj * 2 * sizeof(int));
    int idx = 0;
    for (int c = 0; c < nc; c++) {
        for (int p = 0; p < npjs[c]; p++) {
            coordinates_flat[idx * 2] = coordinates[c][p * 2];       // x
            coordinates_flat[idx * 2 + 1] = coordinates[c][p * 2 + 1]; // y
            idx++;
        }
        free(coordinates[c]);
    }
    free(coordinates);

    // Prepare class IDs
    int *class_ids = (int *)malloc(total_npj * sizeof(int));
    idx = 0;
    for (int c = 0; c < nc; c++) {
        for (int p = 0; p < npjs[c]; p++) {
            class_ids[idx] = c;
            idx++;
        }
    }

    // Allocate device memory and copy data
    int *d_npjs;
    CSC(cudaMalloc(&d_npjs, nc * sizeof(int)));
    CSC(cudaMemcpy(d_npjs, npjs, nc * sizeof(int), cudaMemcpyHostToDevice));

    int *d_class_ids;
    CSC(cudaMalloc(&d_class_ids, total_npj * sizeof(int)));
    CSC(cudaMemcpy(d_class_ids, class_ids, total_npj * sizeof(int), cudaMemcpyHostToDevice));

    int *d_coordinates_flat;
    CSC(cudaMalloc(&d_coordinates_flat, total_npj * 2 * sizeof(int)));
    CSC(cudaMemcpy(d_coordinates_flat, coordinates_flat, total_npj * 2 * sizeof(int), cudaMemcpyHostToDevice));

    float3 *d_sample_pixels;
    CSC(cudaMalloc(&d_sample_pixels, total_npj * sizeof(float3)));

    double total_kernel_time = 0.0f; // Variable to accumulate kernel execution times

    // Read sample pixels
    MEASURE_KERNEL_TIME((read_sample_pixels<<<THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y>>>(d_image, w, h, total_npj, d_coordinates_flat, d_sample_pixels)), total_kernel_time);

    // Compute sums
    double *d_sums_r, *d_sums_g, *d_sums_b;
    CSC(cudaMalloc(&d_sums_r, nc * sizeof(double)));
    CSC(cudaMalloc(&d_sums_g, nc * sizeof(double)));
    CSC(cudaMalloc(&d_sums_b, nc * sizeof(double)));
    CSC(cudaMemset(d_sums_r, 0, nc * sizeof(double)));
    CSC(cudaMemset(d_sums_g, 0, nc * sizeof(double)));
    CSC(cudaMemset(d_sums_b, 0, nc * sizeof(double)));

    MEASURE_KERNEL_TIME((compute_sums<<<THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y>>>(total_npj, d_class_ids, d_sample_pixels, d_sums_r, d_sums_g, d_sums_b)), total_kernel_time);

    // Compute averages
    double *d_avg_r, *d_avg_g, *d_avg_b;
    CSC(cudaMalloc(&d_avg_r, nc * sizeof(double)));
    CSC(cudaMalloc(&d_avg_g, nc * sizeof(double)));
    CSC(cudaMalloc(&d_avg_b, nc * sizeof(double)));

    MEASURE_KERNEL_TIME((compute_averages<<<THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y>>>(nc, d_sums_r, d_sums_g, d_sums_b, d_npjs, d_avg_r, d_avg_g, d_avg_b)), total_kernel_time);

    // Copy averages to host and then to constant memory
    double *h_avg_r = (double *)malloc(nc * sizeof(double));
    double *h_avg_g = (double *)malloc(nc * sizeof(double));
    double *h_avg_b = (double *)malloc(nc * sizeof(double));
    CSC(cudaMemcpy(h_avg_r, d_avg_r, nc * sizeof(double), cudaMemcpyDeviceToHost));
    CSC(cudaMemcpy(h_avg_g, d_avg_g, nc * sizeof(double), cudaMemcpyDeviceToHost));
    CSC(cudaMemcpy(h_avg_b, d_avg_b, nc * sizeof(double), cudaMemcpyDeviceToHost));

    CSC(cudaMemcpyToSymbol(const_avg_r, h_avg_r, nc * sizeof(double)));
    CSC(cudaMemcpyToSymbol(const_avg_g, h_avg_g, nc * sizeof(double)));
    CSC(cudaMemcpyToSymbol(const_avg_b, h_avg_b, nc * sizeof(double)));

    free(h_avg_r);
    free(h_avg_g);
    free(h_avg_b);

    // Compute covariance matrices
    double *d_covariance_matrices;
    CSC(cudaMalloc(&d_covariance_matrices, nc * 9 * sizeof(double)));
    CSC(cudaMemset(d_covariance_matrices, 0, nc * 9 * sizeof(double)));

    MEASURE_KERNEL_TIME((compute_covariances<<<THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y>>>(total_npj, d_class_ids, d_sample_pixels, d_avg_r, d_avg_g, d_avg_b, d_covariance_matrices)), total_kernel_time);

    // Finalize covariance matrices
    MEASURE_KERNEL_TIME((finalize_covariances<<<THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y>>>(nc, d_covariance_matrices, d_npjs)), total_kernel_time);

    // Invert covariance matrices
    double *d_inverse_covariance_matrices;
    CSC(cudaMalloc(&d_inverse_covariance_matrices, nc * 9 * sizeof(double)));
    MEASURE_KERNEL_TIME((invert_covariances<<<THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y>>>(nc, d_covariance_matrices, d_inverse_covariance_matrices)), total_kernel_time);

    // Copy inverse covariance matrices to host and then to constant memory
    double *h_inverse_covariance_matrices = (double *)malloc(nc * 9 * sizeof(double));
    CSC(cudaMemcpy(h_inverse_covariance_matrices, d_inverse_covariance_matrices, nc * 9 * sizeof(double), cudaMemcpyDeviceToHost));

    double h_inv_cov_matrices[MAX_CLASSES][3][3];
    for (int c = 0; c < nc; c++) {
        double *src = &h_inverse_covariance_matrices[c * 9];
        h_inv_cov_matrices[c][0][0] = src[0];
        h_inv_cov_matrices[c][0][1] = src[1];
        h_inv_cov_matrices[c][0][2] = src[2];
        h_inv_cov_matrices[c][1][0] = src[3];
        h_inv_cov_matrices[c][1][1] = src[4];
        h_inv_cov_matrices[c][1][2] = src[5];
        h_inv_cov_matrices[c][2][0] = src[6];
        h_inv_cov_matrices[c][2][1] = src[7];
        h_inv_cov_matrices[c][2][2] = src[8];
    }
    CSC(cudaMemcpyToSymbol(const_inv_covariance_matrices, h_inv_cov_matrices, sizeof(double) * nc * 3 * 3));

    free(h_inverse_covariance_matrices);

    // Run classification kernel
    MEASURE_KERNEL_TIME((classify_kernel<<<dim3(BLOCKS_PER_GRID_X, BLOCKS_PER_GRID_Y), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y)>>>(d_image, w, h, nc)), total_kernel_time);

    // Copy result back to host
    CSC(cudaMemcpy(data, d_image, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    CSC(cudaFree(d_image));

    // printf("CUDA execution time: <%f ms>\n", total_kernel_time);

    fp = fopen(outputFilepath, "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);
    free(data);

    // Free allocated memory
    free(npjs);
    free(offsets);
    free(coordinates_flat);
    free(class_ids);

    CSC(cudaFree(d_npjs));
    CSC(cudaFree(d_class_ids));
    CSC(cudaFree(d_coordinates_flat));
    CSC(cudaFree(d_sample_pixels));
    CSC(cudaFree(d_sums_r));
    CSC(cudaFree(d_sums_g));
    CSC(cudaFree(d_sums_b));
    CSC(cudaFree(d_avg_r));
    CSC(cudaFree(d_avg_g));
    CSC(cudaFree(d_avg_b));
    CSC(cudaFree(d_covariance_matrices));
    CSC(cudaFree(d_inverse_covariance_matrices));

    return 0;
}
