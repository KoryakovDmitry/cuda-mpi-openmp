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
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int y = idy; y < h; y += offsety) {
        for (int x = idx; x < w; x += offsetx) {
            // Handle boundary conditions by clamping coordinates
            int x1 = min(x + 1, w - 1);
            int y1 = min(y + 1, h - 1);

            // Read the pixel values from the texture
            uchar4 w11 = tex2D<uchar4>(tex, x, y);
            uchar4 w12 = tex2D<uchar4>(tex, x1, y);
            uchar4 w21 = tex2D<uchar4>(tex, x, y1);
            uchar4 w22 = tex2D<uchar4>(tex, x1, y1);

            // Compute Gx and Gy for each color channel using the Roberts operator
            int Gx_r = (int)w11.x - (int)w22.x;
            int Gy_r = (int)w12.x - (int)w21.x;
            int Gx_g = (int)w11.y - (int)w22.y;
            int Gy_g = (int)w12.y - (int)w21.y;
            int Gx_b = (int)w11.z - (int)w22.z;
            int Gy_b = (int)w12.z - (int)w21.z;

            // Compute the gradient magnitude using the sum of absolute values
            int G_r = abs(Gx_r) + abs(Gy_r);
            int G_g = abs(Gx_g) + abs(Gy_g);
            int G_b = abs(Gx_b) + abs(Gy_b);

            // Clamp the values to [0, 255]
            G_r = min(max(G_r, 0), 255);
            G_g = min(max(G_g, 0), 255);
            G_b = min(max(G_b, 0), 255);

            // Set the output pixel value with alpha channel set to zero
            out[y * w + x] = make_uchar4(G_r, G_g, G_b, w11.w);
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
