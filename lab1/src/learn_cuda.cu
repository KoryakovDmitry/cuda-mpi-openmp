//
// Created by dm on 09/11/2024.
//
#include <stdio.h>
#include <stdlib.h>
#include <cudaruntime.h>

int main() {
    int i, n = 10000;
    int *arr = (int *)malloc(sizeof(int) * n);
    for (i = 0; i < n; i++) {
        arr[i] = i;
    }

    int *dev_arr;
    cudaMalloc(&dev_arr, sizeof(int) * n);
    cudaMemcpy(dev_arr, arr, sizeof(int) * n, cudaMemcpyHostToDevice);

    cudaFree(dev_arr);
    free(arr);
    return 0;
}
