#include <stdio.h>
#include <stdlib.h>

int main() {
    int nc; // Количество классов

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

    // Вывод для проверки
    for (int c = 0; c < nc; c++) {
        printf("Класс %d:\n", c + 1);
        printf("Количество пикселей: %d\n", npjs[c]);
        printf("Координаты:\n");
        for (int p = 0; p < npjs[c]; p++) {
            printf("(%d, %d)\n", coordinates[c][p * 2], coordinates[c][p * 2 + 1]);
        }
    }

    // Освобождение выделенной памяти
    for (int c = 0; c < nc; c++) {
        free(coordinates[c]);
    }
    free(coordinates);
    free(npjs);

    return 0;
}