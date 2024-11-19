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

// Device function to invert a 3x3 matrix
__device__ void invert_3x3_matrix(const float *a, float *inv_a) {
    // Compute the determinant
    float det = a[0]*(a[4]*a[8] - a[5]*a[7]) - a[1]*(a[3]*a[8] - a[5]*a[6]) + a[2]*(a[3]*a[7] - a[4]*a[6]);

    if (fabsf(det) < 1e-6f) {
        // Matrix is singular, set inverse to zero matrix
        for (int i = 0; i < 9; i++)
            inv_a[i] = 0.0f;
        return;
    }

    float inv_det = 1.0f / det;

    // Compute adjugate matrix
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
__global__ void read_sample_pixels(cudaTextureObject_t tex, int total_npj, int *d_coordinates_flat, float3 *d_sample_pixels) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= total_npj)
        return;

    int x = d_coordinates_flat[tid * 2];
    int y = d_coordinates_flat[tid * 2 + 1];

    uchar4 p = tex2D<uchar4>(tex, x, y);

    d_sample_pixels[tid] = make_float3((float)p.x, (float)p.y, (float)p.z);
}

// Kernel to compute sums for means
__global__ void compute_sums(int total_npj, int *d_class_ids, float3 *d_sample_pixels, float *d_sums_r, float *d_sums_g, float *d_sums_b) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= total_npj)
        return;

    int class_id = d_class_ids[tid];

    float3 p = d_sample_pixels[tid];

    atomicAdd(&d_sums_r[class_id], p.x);
    atomicAdd(&d_sums_g[class_id], p.y);
    atomicAdd(&d_sums_b[class_id], p.z);
}

// Kernel to compute averages
__global__ void compute_averages(int nc, float *d_sums_r, float *d_sums_g, float *d_sums_b, int *d_npjs, float *d_avg_r, float *d_avg_g, float *d_avg_b) {
    int c = threadIdx.x + blockIdx.x * blockDim.x;

    if (c >= nc)
        return;

    int npj = d_npjs[c];

    if (npj > 0) {
        d_avg_r[c] = d_sums_r[c] / npj;
        d_avg_g[c] = d_sums_g[c] / npj;
        d_avg_b[c] = d_sums_b[c] / npj;
    } else {
        d_avg_r[c] = 0.0f;
        d_avg_g[c] = 0.0f;
        d_avg_b[c] = 0.0f;
    }
}

// Kernel to compute covariance matrices
__global__ void compute_covariances(int total_npj, int *d_class_ids, float3 *d_sample_pixels, float *d_avg_r, float *d_avg_g, float *d_avg_b, float *d_covariance_matrices) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid >= total_npj)
        return;

    int class_id = d_class_ids[tid];

    float3 p = d_sample_pixels[tid];

    float3 avg;
    avg.x = d_avg_r[class_id];
    avg.y = d_avg_g[class_id];
    avg.z = d_avg_b[class_id];

    float3 diff;
    diff.x = p.x - avg.x;
    diff.y = p.y - avg.y;
    diff.z = p.z - avg.z;

    // Indices of covariance matrix elements:
    // [0 1 2]
    // [3 4 5]
    // [6 7 8]

    float *cov = &d_covariance_matrices[class_id * 9];

    atomicAdd(&cov[0], diff.x * diff.x);
    atomicAdd(&cov[1], diff.x * diff.y);
    atomicAdd(&cov[2], diff.x * diff.z);

    atomicAdd(&cov[3], diff.y * diff.x);
    atomicAdd(&cov[4], diff.y * diff.y);
    atomicAdd(&cov[5], diff.y * diff.z);

    atomicAdd(&cov[6], diff.z * diff.x);
    atomicAdd(&cov[7], diff.z * diff.y);
    atomicAdd(&cov[8], diff.z * diff.z);
}

// Kernel to finalize covariance matrices
__global__ void finalize_covariances(int nc, float *d_covariance_matrices, int *d_npjs) {
    int c = threadIdx.x + blockIdx.x * blockDim.x;

    if (c >= nc)
        return;

    int npj = d_npjs[c];

    if (npj > 1) {
        float inv_np1 = 1.0f / (npj - 1);

        float *cov = &d_covariance_matrices[c * 9];

        for (int i = 0; i < 9; i++) {
            cov[i] *= inv_np1;
        }
    } else {
        // If npj <= 1, set covariance matrix to identity
        float *cov = &d_covariance_matrices[c * 9];

        for (int i = 0; i < 9; i++) {
            cov[i] = 0.0f;
        }
    }
}

// Kernel to invert covariance matrices
__global__ void invert_covariances(int nc, float *d_covariance_matrices, float *d_inverse_covariance_matrices) {
    int c = threadIdx.x + blockIdx.x * blockDim.x;

    if (c >= nc)
        return;

    float *cov = &d_covariance_matrices[c * 9];
    float *inv_cov = &d_inverse_covariance_matrices[c * 9];

    invert_3x3_matrix(cov, inv_cov);
}

// Main kernel to compute Mahalanobis distances and assign class labels
__global__ void kernel(cudaTextureObject_t tex, uchar4 *out, int w, int h, int nc, float *d_avg_r, float *d_avg_g, float *d_avg_b, float *d_inverse_covariance_matrices) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int y = idy; y < h; y += offsety) {
        for (int x = idx; x < w; x += offsetx) {
            // Read pixel values from the texture
            uchar4 p_uchar = tex2D<uchar4>(tex, x, y);
            float3 p;
            p.x = (float)p_uchar.x;
            p.y = (float)p_uchar.y;
            p.z = (float)p_uchar.z;

            float min_m = FLT_MAX;
            int label_class_idx_int = -1;

            for (int c = 0; c < nc; c++) {
                float3 avg_j;
                avg_j.x = d_avg_r[c];
                avg_j.y = d_avg_g[c];
                avg_j.z = d_avg_b[c];

                float3 diff;
                diff.x = p.x - avg_j.x;
                diff.y = p.y - avg_j.y;
                diff.z = p.z - avg_j.z;

                float *inv_cov = &d_inverse_covariance_matrices[c * 9];

                float3 temp;
                temp.x = inv_cov[0] * diff.x + inv_cov[1] * diff.y + inv_cov[2] * diff.z;
                temp.y = inv_cov[3] * diff.x + inv_cov[4] * diff.y + inv_cov[5] * diff.z;
                temp.z = inv_cov[6] * diff.x + inv_cov[7] * diff.y + inv_cov[8] * diff.z;

                float m = diff.x * temp.x + diff.y * temp.y + diff.z * temp.z;

                if (m < min_m) {
                    min_m = m;
                    label_class_idx_int = c;
                } else if (m == min_m && c < label_class_idx_int) {
                    label_class_idx_int = c;
                }
            }

            // Convert to unsigned char
            unsigned char label_class = static_cast<unsigned char>(label_class_idx_int);

            // Set the output pixel
            out[y * w + x] = make_uchar4(p_uchar.x, p_uchar.y, p_uchar.z, label_class);
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
        printf("Memory allocation error!\n");
        return 1;
    }

    // Arrays to store the coordinates of pixels (2D dynamic array)
    int **coordinates = (int **)malloc(nc * sizeof(int *));
    if (coordinates == NULL) {
        printf("Memory allocation error!\n");
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
            printf("Memory allocation error!\n");
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

    cudaArray *arr;
    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    uchar4 *dev_out;
    CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

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

    float total_kernel_time = 0.0f; // Variable to accumulate kernel execution times

    // Read sample pixels
    int threadsPerBlock = 256;
    int blocksPerGrid = 256;
    MEASURE_KERNEL_TIME((read_sample_pixels<<<blocksPerGrid, threadsPerBlock>>>(tex, total_npj, d_coordinates_flat, d_sample_pixels)), total_kernel_time);

    // Compute sums
    float *d_sums_r, *d_sums_g, *d_sums_b;
    CSC(cudaMalloc(&d_sums_r, nc * sizeof(float)));
    CSC(cudaMalloc(&d_sums_g, nc * sizeof(float)));
    CSC(cudaMalloc(&d_sums_b, nc * sizeof(float)));
    CSC(cudaMemset(d_sums_r, 0, nc * sizeof(float)));
    CSC(cudaMemset(d_sums_g, 0, nc * sizeof(float)));
    CSC(cudaMemset(d_sums_b, 0, nc * sizeof(float)));

    MEASURE_KERNEL_TIME((compute_sums<<<blocksPerGrid, threadsPerBlock>>>(total_npj, d_class_ids, d_sample_pixels, d_sums_r, d_sums_g, d_sums_b)), total_kernel_time);

    // Compute averages
    float *d_avg_r, *d_avg_g, *d_avg_b;
    CSC(cudaMalloc(&d_avg_r, nc * sizeof(float)));
    CSC(cudaMalloc(&d_avg_g, nc * sizeof(float)));
    CSC(cudaMalloc(&d_avg_b, nc * sizeof(float)));

    MEASURE_KERNEL_TIME((compute_averages<<<blocksPerGrid, threadsPerBlock>>>(nc, d_sums_r, d_sums_g, d_sums_b, d_npjs, d_avg_r, d_avg_g, d_avg_b)), total_kernel_time);

    // Compute covariance matrices
    float *d_covariance_matrices;
    CSC(cudaMalloc(&d_covariance_matrices, nc * 9 * sizeof(float)));
    CSC(cudaMemset(d_covariance_matrices, 0, nc * 9 * sizeof(float)));

    MEASURE_KERNEL_TIME((compute_covariances<<<blocksPerGrid, threadsPerBlock>>>(total_npj, d_class_ids, d_sample_pixels, d_avg_r, d_avg_g, d_avg_b, d_covariance_matrices)), total_kernel_time);

    // Finalize covariance matrices
    MEASURE_KERNEL_TIME((finalize_covariances<<<blocksPerGrid, threadsPerBlock>>>(nc, d_covariance_matrices, d_npjs)), total_kernel_time);

    // Invert covariance matrices
    float *d_inverse_covariance_matrices;
    CSC(cudaMalloc(&d_inverse_covariance_matrices, nc * 9 * sizeof(float)));
    MEASURE_KERNEL_TIME((invert_covariances<<<blocksPerGrid, threadsPerBlock>>>(nc, d_covariance_matrices, d_inverse_covariance_matrices)), total_kernel_time);

    // Launch the main kernel
    const int BLOCK_SIZE_X = 32;
    const int BLOCK_SIZE_Y = 32;
    const int GRID_SIZE_X = 16;
    const int GRID_SIZE_Y = 16;
    MEASURE_KERNEL_TIME((kernel<<<dim3(GRID_SIZE_X, GRID_SIZE_Y), dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y)>>>(tex, dev_out, w, h, nc, d_avg_r, d_avg_g, d_avg_b, d_inverse_covariance_matrices)), total_kernel_time);

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

//     printf("CUDA execution time: <%f ms>\n", total_kernel_time);

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
