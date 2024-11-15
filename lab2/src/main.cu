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
    int x, y;
    uchar4 p;
    for(y = idy; y < h; y += offsety
		for(x = idx; x < w; x += offsetx) {
            p = tex2D<uchar4>(tex, x / w, y / h);
            // YOUR CODE. TASK: Выделение контуров. Метод Робертса.
            // out[y * w + x] = // YOUR CODE
            out[y * w + x] = make_uchar4(255 - p.x, 255 - p.y, 255 - p.z, p.w);
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
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = true;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    uchar4 *dev_out;
	CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

    cudaEvent_t start, stop;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));

    CSC(cudaEventRecord(start));
    kernel<<<dim3(16, 16), dim3(32, 32)>>>(tex, dev_out, w, h);
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
    return 0;
}
