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
            // Read pixel values from the texture
            x = max(min(x, w), 1)
            y = max(min(x, h), 1)
            uchar4 p00 = tex2D<uchar4>(tex, x, y);
            uchar4 p10 = tex2D<uchar4>(tex, x + 1, y);
            uchar4 p01 = tex2D<uchar4>(tex, x, y + 1);
            uchar4 p11 = tex2D<uchar4>(tex, x + 1, y + 1);

            // Convert RGB to luminance (grayscale)
            float Y00 = 0.299f * p00.x + 0.587f * p00.y + 0.114f * p00.z;
            float Y10 = 0.299f * p10.x + 0.587f * p10.y + 0.114f * p10.z;
            float Y01 = 0.299f * p01.x + 0.587f * p01.y + 0.114f * p01.z;
            float Y11 = 0.299f * p11.x + 0.587f * p11.y + 0.114f * p11.z;

            // Apply the Roberts operator
            float Gx = Y11 - Y00;   // Gradient in x-direction
            float Gy = Y10 - Y01;   // Gradient in y-direction

            // Calculate the gradient magnitude
            float G = sqrtf(Gx * Gx + Gy * Gy);

            // Clamp the result to [0, 255]
            // G = fminf(fmaxf(G, 0.0f), 255.0f);

            // Convert to unsigned char
            unsigned char res = static_cast<unsigned char>(G);

            // Set the output pixel
            out[y * w + x] = make_uchar4(res, res, res, p00.w);
        }
    }
}

int main() {
    int w, h;

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
    const int BLOCK_SIZE_X = 32;
    const int BLOCK_SIZE_Y = 32;
    const int GRID_SIZE_X = 16;
    const int GRID_SIZE_Y = 16;
    kernel<<<dim3(GRID_SIZE_X, GRID_SIZE_Y), dim3(BLOCK_SIZE_X, BLOCK_SIZE_Y)>>>(tex, dev_out, w, h);
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

//     printf("CUDA execution time: %f ms\n", t);

    fp = fopen(outputFilepath, "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);
    free(data);
    return 0;
}
