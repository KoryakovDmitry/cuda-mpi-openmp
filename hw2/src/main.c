#include <stdio.h>
#include <stdlib.h>

void bub_sort(float *arr, int n) {
    int i, j;
    for (i = 0; i < n - 1; i++) {
        for (j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                float temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

int main() {
    int n;
    scanf("%d", &n);

    float *arr = (float *)malloc(n * sizeof(float));
    if (arr == NULL) {
        printf("Ошибка выделения памяти\n");
        return 1;
    }

    int i;
    for (i = 0; i < n; i++) {
        scanf("%f", &arr[i]);
    }

    bub_sort(arr, n);

    for (i = 0; i < n; i++) {
        printf("%.6e ", arr[i]);
    }
    printf("\n");

    free(arr);

    return 0;
}
