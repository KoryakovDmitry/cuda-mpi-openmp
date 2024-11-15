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

__global__ void kernel(cudaTextureObject_t tex, uchar4 *out, int w, int h) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int j = y; j < h - 1; j += offsety) {  // Adjusted loop bounds
        for (int i = x; i < w - 1; i += offsetx) {
            // Read the pixel values from the texture
            uchar4 w11 = tex2D<uchar4>(tex, i, j);
            uchar4 w12 = tex2D<uchar4>(tex, i + 1, j);
            uchar4 w21 = tex2D<uchar4>(tex, i, j + 1);
            uchar4 w22 = tex2D<uchar4>(tex, i + 1, j + 1);

            // Compute Gx and Gy for each color channel
            int Gx_r = (int)w22.x - (int)w11.x;
            int Gy_r = (int)w21.x - (int)w12.x;
            int Gx_g = (int)w22.y - (int)w11.y;
            int Gy_g = (int)w21.y - (int)w12.y;
            int Gx_b = (int)w22.z - (int)w11.z;
            int Gy_b = (int)w21.z - (int)w12.z;

            // Compute the gradient magnitude for each channel
            int G_r = (int)sqrtf(Gx_r * Gx_r + Gy_r * Gy_r);
            int G_g = (int)sqrtf(Gx_g * Gx_g + Gy_g * Gy_g);
            int G_b = (int)sqrtf(Gx_b * Gx_b + Gy_b * Gy_b);

            // Clamp the values to [0, 255]
            G_r = min(max(G_r, 0), 255);
            G_g = min(max(G_g, 0), 255);
            G_b = min(max(G_b, 0), 255);

            // Set the output pixel value
            out[j * w + i] = make_uchar4(G_r, G_g, G_b, w11.w);
        }
    }
}

int main() {
    int w, h, block_size_x, block_size_y, grid_size_x, grid_size_y;

    // set block_size_x
    scanf("%d", &block_size_x);
    // set block_size_y
    scanf("%d", &block_size_y);
    // set grid_size_x
    scanf("%d", &grid_size_x);
    // set grid_size_y
    scanf("%d", &grid_size_y);

    char inputFilepath[1024], outputFilepath[1024];
    scanf("%1024s", inputFilepath);
    scanf("%1024s", outputFilepath);

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

    cudaEvent_t start, stop;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));

    CSC(cudaEventRecord(start));
    kernel<<<dim3(grid_size_x, grid_size_y), dim3(block_size_x, block_size_y)>>>(tex, dev_out, w, h);
    CSC(cudaEventRecord(stop));
    CSC(cudaEventSynchronize(stop));
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

    float t;
    CSC(cudaEventElapsedTime(&t, start, stop));
    CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(stop));

    printf("CUDA execution time: <%f ms>\n", t);

    fp = fopen(outputFilepath, "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);
    free(data);
    printf("FINISHED!\n");
    return 0;
}
