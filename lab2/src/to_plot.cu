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
            uchar4 p = tex2D<uchar4>(tex, x, y);
            uchar4 px = tex2D<uchar4>(tex, x + 1, y);
            uchar4 py = tex2D<uchar4>(tex, x, y + 1);
            uchar4 pxy = tex2D<uchar4>(tex, x + 1, y + 1);

            int gx_r = abs(p.x - pxy.x) + abs(px.x - py.x);
            int gx_g = abs(p.y - pxy.y) + abs(px.y - py.y);
            int gx_b = abs(p.z - pxy.z) + abs(px.z - py.z);

            uchar4 result;
            result.x = min(gx_r, 255);  // Clamp the result to 255
            result.y = min(gx_g, 255);
            result.z = min(gx_b, 255);
            result.w = 255;  // Alpha channel remains fully opaque

            out[y * w + x] = result;
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

    printf("CUDA execution time: %f ms\n", t);

    fp = fopen(outputFilepath, "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);
    free(data);
    printf("FINISHED!", t);
    return 0;
}
