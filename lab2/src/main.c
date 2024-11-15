#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

typedef struct {
    unsigned char x; // R
    unsigned char y; // G
    unsigned char z; // B
    unsigned char w; // A
} uchar4;

uchar4 getPixel(const uchar4 *data, int w, int h, int x, int y) {
    // Clamp coordinates to valid range
    if (x < 0) x = 0;
    if (x >= w) x = w - 1;
    if (y < 0) y = 0;
    if (y >= h) y = h - 1;
    return data[y * w + x];
}

void processImage(const uchar4 *data, uchar4 *out, int w, int h) {
    int x, y;
    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {
            uchar4 p00 = getPixel(data, w, h, x, y);
            uchar4 p10 = getPixel(data, w, h, x + 1, y);
            uchar4 p01 = getPixel(data, w, h, x, y + 1);
            uchar4 p11 = getPixel(data, w, h, x + 1, y + 1);

            // Convert RGB to luminance (grayscale)
            float Y00 = 0.299f * p00.x + 0.587f * p00.y + 0.114f * p00.z;
            float Y10 = 0.299f * p10.x + 0.587f * p10.y + 0.114f * p10.z;
            float Y01 = 0.299f * p01.x + 0.587f * p01.y + 0.114f * p01.z;
            float Y11 = 0.299f * p11.x + 0.587f * p11.y + 0.114f * p11.z;

            // Apply the Roberts operator
            float Gx = Y11 - Y00;   // Gradient in x-direction
            float Gy = Y10 - Y01;   // Gradient in y-direction

            // Calculate the gradient magnitude
            float G = sqrt(Gx * Gx + Gy * Gy);

            // Clamp the result to [0, 255]
            if (G < 0.0f) G = 0.0f;
            if (G > 255.0f) G = 255.0f;

            // Convert to unsigned char
            unsigned char res = (unsigned char)G;

            // Set the output pixel
            out[y * w + x].x = res;
            out[y * w + x].y = res;
            out[y * w + x].z = res;
            out[y * w + x].w = p00.w; // Preserve alpha channel
        }
    }
}

int main() {
    int w, h;
    char inputFilepath[1024], outputFilepath[1024];
    scanf("%1023s", inputFilepath);
    scanf("%1023s", outputFilepath);

    FILE *fp = fopen(inputFilepath, "rb");
    if (fp == NULL) {
        fprintf(stderr, "Error opening input file.\n");
        return 1;
    }

    if (fread(&w, sizeof(int), 1, fp) != 1 || fread(&h, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "Error reading image dimensions.\n");
        fclose(fp);
        return 1;
    }

    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    if (data == NULL) {
        fprintf(stderr, "Error allocating memory for input image.\n");
        fclose(fp);
        return 1;
    }

    if (fread(data, sizeof(uchar4), w * h, fp) != w * h) {
        fprintf(stderr, "Error reading image data.\n");
        free(data);
        fclose(fp);
        return 1;
    }
    fclose(fp);

    uchar4 *out = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    if (out == NULL) {
        fprintf(stderr, "Error allocating memory for output image.\n");
        free(data);
        return 1;
    }

    clock_t time_cpu_start = clock();
    processImage(data, out, w, h);
    clock_t time_cpu_end = clock();
    double cpu_time = ((double)(time_cpu_end - time_cpu_start)) / CLOCKS_PER_SEC * 1000;
    // Uncomment the following line to display execution time
     printf("CPU execution time: <%f ms>\n", cpu_time);

    fp = fopen(outputFilepath, "wb");
    if (fp == NULL) {
        fprintf(stderr, "Error opening output file.\n");
        free(data);
        free(out);
        return 1;
    }

    if (fwrite(&w, sizeof(int), 1, fp) != 1 || fwrite(&h, sizeof(int), 1, fp) != 1) {
        fprintf(stderr, "Error writing image dimensions.\n");
        free(data);
        free(out);
        fclose(fp);
        return 1;
    }

    if (fwrite(out, sizeof(uchar4), w * h, fp) != w * h) {
        fprintf(stderr, "Error writing image data.\n");
        free(data);
        free(out);
        fclose(fp);
        return 1;
    }
    fclose(fp);

    free(data);
    free(out);

    return 0;
}
