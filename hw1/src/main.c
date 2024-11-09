#include <stdio.h>
#include <math.h>

int main() {
    float a, b, c;
    scanf("%f %f %f", &a, &b, &c);
    
    if (a == 0) {
        if (b == 0) {
            if (c == 0) {
                printf("any\n");
            } else {
                printf("incorrect\n");
            }
        } else {
            float root = -c / b;
            printf("%.6f\n", root);
        }
    } else {
        float D = b * b - 4 * a * c;
        if (D > 0) {
            float sqrtD = sqrtf(D);
            float root1 = (-b + sqrtD) / (2 * a);
            float root2 = (-b - sqrtD) / (2 * a);
            printf("%.6f %.6f\n", root1, root2);
        } else if (D == 0) {
            float root = -b / (2 * a);
            printf("%.6f\n", root);
        } else {
            printf("imaginary\n");
        }
    }
    
    return 0;
} 
