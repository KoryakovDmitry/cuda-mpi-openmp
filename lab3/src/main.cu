#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
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

// Kernel for classifying pixels
__global__ void classify_kernel(uchar4 *d_image, int w, int h, int nc) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = w * h;
    if (idx >= total_pixels) return;

    uchar4 pixel = d_image[idx];
    double p_r = (double)pixel.x;
    double p_g = (double)pixel.y;
    double p_b = (double)pixel.z;

    double min_distance = DBL_MAX;
    int best_class = 0;

    for (int j = 0; j < nc; j++) {
        double delta_r = p_r - const_avg[j].x;
        double delta_g = p_g - const_avg[j].y;
        double delta_b = p_b - const_avg[j].z;

        // Compute Mahalanobis distance: delta^T * inv_cov * delta
        double distance =
            delta_r * const_inv_covariance[j][0][0] * delta_r +
            delta_r * const_inv_covariance[j][0][1] * delta_g +
            delta_r * const_inv_covariance[j][0][2] * delta_b +
            delta_g * const_inv_covariance[j][1][0] * delta_r +
            delta_g * const_inv_covariance[j][1][1] * delta_g +
            delta_g * const_inv_covariance[j][1][2] * delta_b +
            delta_b * const_inv_covariance[j][2][0] * delta_r +
            delta_b * const_inv_covariance[j][2][1] * delta_g +
            delta_b * const_inv_covariance[j][2][2] * delta_b;

        if (distance < min_distance) {
            min_distance = distance;
            best_class = j;
        }
    }

    // Assign the class index to the alpha channel
    d_image[idx].w = (unsigned char)best_class;
}

// Host struct for double3
typedef struct {
    double x;
    double y;
    double z;
} double3_host;

// Function to invert a 3x3 matrix
int invert_matrix(double input[3][3], double output[3][3]) {
    double det = input[0][0]*(input[1][1]*input[2][2] - input[1][2]*input[2][1]) -
                 input[0][1]*(input[1][0]*input[2][2] - input[1][2]*input[2][0]) +
                 input[0][2]*(input[1][0]*input[2][1] - input[1][1]*input[2][0]);

    if (fabs(det) < 1e-12) {
        return 0; // Singular matrix
    }

    double inv_det = 1.0 / det;

    output[0][0] =  (input[1][1]*input[2][2] - input[1][2]*input[2][1]) * inv_det;
    output[0][1] = -(input[0][1]*input[2][2] - input[0][2]*input[2][1]) * inv_det;
    output[0][2] =  (input[0][1]*input[1][2] - input[0][2]*input[1][1]) * inv_det;

    output[1][0] = -(input[1][0]*input[2][2] - input[1][2]*input[2][0]) * inv_det;
    output[1][1] =  (input[0][0]*input[2][2] - input[0][2]*input[2][0]) * inv_det;
    output[1][2] = -(input[0][0]*input[1][2] - input[0][2]*input[1][0]) * inv_det;

    output[2][0] =  (input[1][0]*input[2][1] - input[1][1]*input[2][0]) * inv_det;
    output[2][1] = -(input[0][0]*input[2][1] - input[0][1]*input[2][0]) * inv_det;
    output[2][2] =  (input[0][0]*input[1][1] - input[0][1]*input[1][0]) * inv_det;

    return 1; // Inversion successful
}

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
    if (!fin) {
        fprintf(stderr, "Error opening input file.\n");
        return 1;
    }
    if (fread(&w, sizeof(int), 1, fin) != 1 || fread(&h, sizeof(int), 1, fin) != 1) {
        fprintf(stderr, "Error reading image dimensions.\n");
        fclose(fin);
        return 1;
    }

    // Allocate host and device memory for the image
    long long total_pixels_ll = (long long)w * (long long)h;
    if (total_pixels_ll > 400000000LL) { // 4 * 10^8
        fprintf(stderr, "Image size exceeds the maximum allowed.\n");
        fclose(fin);
        return 1;
    }
    int total_pixels = (int)total_pixels_ll;

    uchar4 *h_image = (uchar4 *)malloc(total_pixels * sizeof(uchar4));
    if (!h_image) {
        fprintf(stderr, "Memory allocation failed for host image.\n");
        fclose(fin);
        return 1;
    }
    uchar4 *d_image;
    CSC(cudaMalloc(&d_image, total_pixels * sizeof(uchar4)));

    // Read image data
    size_t read_count = fread(h_image, sizeof(uchar4), total_pixels, fin);
    if (read_count != (size_t)total_pixels) {
        fprintf(stderr, "Error reading image data.\n");
        free(h_image);
        fclose(fin);
        return 1;
    }
    fclose(fin);
    CSC(cudaMemcpy(d_image, h_image, total_pixels * sizeof(uchar4), cudaMemcpyHostToDevice));

    // Number of classes
    int nc;
    if (scanf("%d", &nc) != 1) {
        fprintf(stderr, "Error reading number of classes.\n");
        free(h_image);
        CSC(cudaFree(d_image));
        return 1;
    }
    if (nc < 1 || nc > MAX_CLASSES) {
        fprintf(stderr, "Number of classes out of bounds (1 <= nc <= %d).\n", MAX_CLASSES);
        free(h_image);
        CSC(cudaFree(d_image));
        return 1;
    }

    // Host memory for averages and inverse covariances
    double3_host avg_host[MAX_CLASSES];
    double inv_covariance_host[MAX_CLASSES][3][3];

    for (int j = 0; j < nc; j++) {
        int npj;
        if (scanf("%d", &npj) != 1) {
            fprintf(stderr, "Error reading number of pixels for class %d.\n", j);
            free(h_image);
            CSC(cudaFree(d_image));
            return 1;
        }
        if (npj < 1 || npj > (1 << 19)) {
            fprintf(stderr, "Number of pixels in class %d out of bounds (1 <= npj <= 2^19).\n", j);
            free(h_image);
            CSC(cudaFree(d_image));
            return 1;
        }

        // Initialize sums for average
        double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;

        // Temporary arrays to store pixel values for covariance computation
        double *pixels_r = (double *)malloc(npj * sizeof(double));
        double *pixels_g = (double *)malloc(npj * sizeof(double));
        double *pixels_b = (double *)malloc(npj * sizeof(double));
        if (!pixels_r || !pixels_g || !pixels_b) {
            fprintf(stderr, "Memory allocation failed for class %d.\n", j);
            free(h_image);
            CSC(cudaFree(d_image));
            if (pixels_r) free(pixels_r);
            if (pixels_g) free(pixels_g);
            if (pixels_b) free(pixels_b);
            return 1;
        }

        for (int p = 0; p < npj; p++) {
            int x, y;
            if (scanf("%d %d", &x, &y) != 2) {
                fprintf(stderr, "Error reading pixel coordinates for class %d.\n", j);
                free(h_image);
                CSC(cudaFree(d_image));
                free(pixels_r);
                free(pixels_g);
                free(pixels_b);
                return 1;
            }
            if (x < 0 || x >= w || y < 0 || y >= h) {
                fprintf(stderr, "Pixel coordinates out of bounds for class %d: (%d, %d).\n", j, x, y);
                free(h_image);
                CSC(cudaFree(d_image));
                free(pixels_r);
                free(pixels_g);
                free(pixels_b);
                return 1;
            }
            int idx = y * w + x;
            double r = (double)h_image[idx].x;
            double g = (double)h_image[idx].y;
            double b = (double)h_image[idx].z;

            pixels_r[p] = r;
            pixels_g[p] = g;
            pixels_b[p] = b;

            sum_r += r;
            sum_g += g;
            sum_b += b;
        }

        // Compute averages
        avg_host[j].x = sum_r / npj;
        avg_host[j].y = sum_g / npj;
        avg_host[j].z = sum_b / npj;

        // Compute covariance matrix
        double cov[3][3] = {0};
        for (int p = 0; p < npj; p++) {
            double dr = pixels_r[p] - avg_host[j].x;
            double dg = pixels_g[p] - avg_host[j].y;
            double db = pixels_b[p] - avg_host[j].z;

            cov[0][0] += dr * dr;
            cov[0][1] += dr * dg;
            cov[0][2] += dr * db;
            cov[1][0] += dg * dr;
            cov[1][1] += dg * dg;
            cov[1][2] += dg * db;
            cov[2][0] += db * dr;
            cov[2][1] += db * dg;
            cov[2][2] += db * db;
        }

        // Normalize covariance matrix
        if (npj > 1) {
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    cov[a][b] /= (npj - 1);
                }
            }
        } else {
            // If npj == 1, covariance is zero matrix
            memset(cov, 0, 9 * sizeof(double));
        }

        // Invert covariance matrix
        if (!invert_matrix(cov, inv_covariance_host[j])) {
            fprintf(stderr, "Covariance matrix for class %d is singular and cannot be inverted.\n", j);
            free(h_image);
            CSC(cudaFree(d_image));
            free(pixels_r);
            free(pixels_g);
            free(pixels_b);
            return 1;
        }

        // Free temporary pixel arrays
        free(pixels_r);
        free(pixels_g);
        free(pixels_b);
    }

    // Copy averages and inverse covariances to device
    CSC(cudaMemcpyToSymbol(const_avg, avg_host, nc * sizeof(double3_host)));
    CSC(cudaMemcpyToSymbol(const_inv_covariance, inv_covariance_host, nc * sizeof(double[3][3])));

    // Launch classification kernel
    float total_kernel_time = 0.0f;
    MEASURE_KERNEL_TIME((classify_kernel<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_image, w, h, nc)), total_kernel_time);
    CSC(cudaDeviceSynchronize());

    // Copy results back to host and write output
    CSC(cudaMemcpy(h_image, d_image, total_pixels * sizeof(uchar4), cudaMemcpyDeviceToHost));
    FILE *fout = fopen(output_file, "wb");
    if (!fout) {
        fprintf(stderr, "Error opening output file.\n");
        free(h_image);
        CSC(cudaFree(d_image));
        return 1;
    }
    if (fwrite(&w, sizeof(int), 1, fout) != 1 || fwrite(&h, sizeof(int), 1, fout) != 1) {
        fprintf(stderr, "Error writing image dimensions to output file.\n");
        free(h_image);
        CSC(cudaFree(d_image));
        fclose(fout);
        return 1;
    }
    size_t write_count = fwrite(h_image, sizeof(uchar4), total_pixels, fout);
    if (write_count != (size_t)total_pixels) {
        fprintf(stderr, "Error writing image data to output file.\n");
        free(h_image);
        CSC(cudaFree(d_image));
        fclose(fout);
        return 1;
    }
    fclose(fout);

    // Free memory
    free(h_image);
    CSC(cudaFree(d_image));
    // printf("CUDA execution time: <%f ms>\n", total_kernel_time);
    return 0;
}
