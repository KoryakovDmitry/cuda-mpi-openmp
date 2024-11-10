//
// Created by dm on 09/11/2024.
//

#include <stdio.h>
#include <stdlib.h>

int main() {
    // set dim of vector
    int n;
    scanf("%d", &n);
    
    // set mem for  first vector
    double *arr_1 = (double *)malloc(n * sizeof(double));
    if (arr_1 == NULL) {
        printf("Ошибка выделения памяти для `arr_1`\n");
        return 1;
    }
    
    // fill first vector
    int i;
    for (i = 0; i < n; i++) {
        scanf("%lf", &arr_1[i]);
    }
    
    // set mem for first vector
    double *arr_2 = (double *)malloc(n * sizeof(double));
    if (arr_2 == NULL) {
        printf("Ошибка выделения памяти для `arr_2`\n");
        return 1;
    }
    
    // fill second vector
    for (i = 0; i < n; i++) {
        scanf("%lf", &arr_2[i]);
    }


    // set mem for first vector
    double *arr_3 = (double *)malloc(n * sizeof(double));
    if (arr_3 == NULL) {
        printf("Ошибка выделения памяти для `arr_3`\n");
        return 1;
    }

    // calc diff
    for (i = 0; i < n; i++) {
        arr_3[i] = arr_1[i] - arr_2[i];
        printf("%.10e ", arr_3[i]);
    }

    free(arr_1);
    free(arr_2);
    free(arr_3);

    return 0;
}