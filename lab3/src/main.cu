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


// Ядро для вычисления средних для всех классов
// __global__ void compute_means_all_classes(YOUR ARGS) { YOUR CODE }

// Ядро для вычисления ковариационных матриц для всех классов
// __global__ void compute_covariances_all_classes(YOUR ARGS) { YOUR CODE }

__global__ void kernel(cudaTextureObject_t tex, uchar4 *out, int w, int h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    for (int y = idy; y < h; y += offsety) {
        for (int x = idx; x < w; x += offsetx) {
            // Read pixel values from the texture
            uchar4 p = tex2D<uchar4>(tex, x, y);

            // YOUR CODE for Mahalanobis distance
            // j_c = \arg\max_j \left[ -\left( \mathbf{p} - \mathbf{avg}_j \right)^\top \mathbf{cov}_j^{-1} \left( \mathbf{p} - \mathbf{avg}_j \right) \right]

            // Convert to unsigned char
            unsigned char label_class = static_cast<unsigned char>(label_class_idx_int);

            // Set the output pixel
            out[y * w + x] = make_uchar4(p.x, p.y, p.z, label_class);
        }
    }
}


int main() {
    int w, h;
    int nc; // Количество классов

    char inputFilepath[1024], outputFilepath[1024];
    scanf("%1024s", inputFilepath);
    scanf("%1024s", outputFilepath);

    // Чтение количества классов
    scanf("%d", &nc);

    // Выделение памяти для хранения количества пикселей в каждом классе
    int *npjs = (int *)malloc(nc * sizeof(int));
    if (npjs == NULL) {
        printf("Ошибка выделения памяти!\n");
        return 1;
    }

    // Массив для хранения координат пикселей (двумерный динамический массив)
    int **coordinates = (int **)malloc(nc * sizeof(int *));
    if (coordinates == NULL) {
        printf("Ошибка выделения памяти!\n");
        free(npjs);
        return 1;
    }

    // Чтение данных для каждого класса
    for (int c = 0; c < nc; c++) {
        // Чтение количества пикселей
        scanf("%d", &npjs[c]);

        // Выделение памяти для хранения координат (npjs[c] пар чисел)
        coordinates[c] = (int *)malloc(npjs[c] * 2 * sizeof(int));
        if (coordinates[c] == NULL) {
            printf("Ошибка выделения памяти!\n");
            for (int i = 0; i < c; i++) {
                free(coordinates[i]);
            }
            free(coordinates);
            free(npjs);
            return 1;
        }

        // Чтение координат пикселей
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
