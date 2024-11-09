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

    // calc diff
    for (i = 0; i < n; i++) {
        double diff;
        diff = arr_1[i] - arr_2[i];
        printf("%.10e ", diff);
    }
    return 0;
}