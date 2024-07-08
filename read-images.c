#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <stdio.h>

int main() {
    int width, height, channels;
    unsigned char *img = stbi_load("image.jpg", &width, &height, &channels, 0);
    if (img == NULL) {
        printf("Failed to load image\n");
        return 1;
    }
    
    // Use the image data...
    printf("Loaded image with a width of %d px, a height of %d px, and %d channels\n", width, height, channels);
    
    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            printf("%d",img[i*width+j]);
        }
        printf("\n");
    }
    // Free the image memory
    stbi_image_free(img);
    return 0;
}

