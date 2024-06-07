// Importing Standard Libraries
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

// Define model parameters
#define INPUT_NEURONS 784
#define OUPUT_NEURONS 10
#define ACTIVATION "softmax"

// Defin model hyperparameters
#define LEARNING_RATE 0.1
#define EPOCHS 1000

// Dataset paths
#define TRAIN_IMAGES "data/train-images.idx3-ubyte"
#define TRAIN_LABELS "data/train-labels.idx1-ubyte"
#define TEST_IMAGES "data/t10k-images.idx3-ubyte"
#define TEST_LABELS "data/t10k-labels.idx1-ubyte"

// Define customs data types
typedef struct{
    float weights[OUPUT_NEURONS][INPUT_NEURONS];
    float bias[OUPUT_NEURONS];
    void (*activation)(float*,float* size_t);
    void (*D_activation)(float*,float* size_t);
} NeuralNetwork;

// Initialize network
void initialize(NeuralNetwork *nn){
    for (int i = 0; i < OUPUT_NEURONS; i++){
        for (int j=0; j < INPUT_NEURONS; j++){
            nn->weights[i][j] = ((float)rand() / RAND_MAX) * 2 - 1;
        }
         nn->bias[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }
}

// Loss function and its derivative
// Function to calculate the softmax
void softmax(float *input, float *output, int length) {
    float sum = 0.0;
    for (int i = 0; i < length; i++) {
        output[i] = exp(input[i]);
        sum += output[i];
    }

    for (int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}

// Function to calculate the gradient of the cross-entropy loss with softmax
void softmax_cross_entropy_loss_derivative(double *softmax_output, int *target, double *derivative, int length) {
    for (int i = 0; i < length; i++) {
        derivative[i] = softmax_output[i] - target[i];
    }
}


// Function to calculate cross-entropy loss
double cross_entropy_loss(double *softmax_output, int *target, int length) {
    double loss = 0.0;
    for (int i = 0; i < length; i++) {
        if (target[i] == 1) {
            loss -= log(softmax_output[i]);
        }
    }
    return loss;
}

uint8_t* read_mnist_images(const char* filepath, int* number_of_images, int* image_size) {
    FILE* file = fopen(filepath, "rb");
    if (file == NULL) {
        perror("Failed to open file");
        exit(1);
    }

    int magic_number = 0;
    fread(&magic_number, sizeof(int), 1, file);
    magic_number =  __builtin_bswap32(magic_number); // MNIST files are big-endian

    fread(number_of_images, sizeof(int), 1, file);
    *number_of_images = __builtin_bswap32(*number_of_images);

    int rows = 0, cols = 0;
    fread(&rows, sizeof(int), 1, file);
    fread(&cols, sizeof(int), 1, file);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);
    *image_size = rows * cols;

    uint8_t* data = (uint8_t*)malloc(*number_of_images * *image_size);
    fread(data, sizeof(uint8_t), *number_of_images * *image_size, file);
    fclose(file);

    return data;
}

unsigned char* read_mnist_labels(const char* filepath, int* number_of_labels) {
    FILE* file = fopen(filepath, "rb");
    if (file == NULL) {
        perror("Failed to open file");
        exit(1);
    }

    int magic_number = 0;
    fread(&magic_number, sizeof(int), 1, file);
    magic_number = __builtin_bswap32(magic_number); // MNIST files are big-endian

    fread(number_of_labels, sizeof(int), 1, file);
    *number_of_labels = __builtin_bswap32(*number_of_labels);

    unsigned char* data = (unsigned char*)malloc(*number_of_labels);
    fread(data, sizeof(unsigned char), *number_of_labels, file);
    fclose(file);

    return data;
}



void save_ppm_image(const char* filename, uint8_t* image, int width, int height) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file for writing");
        exit(1);
    }

    // Write PPM header
    fprintf(file, "P6\n%d %d\n255\n", width, height);

    // Write pixel data
    for (int i = 0; i < width * height; ++i) {
        unsigned char pixel = image[i];
        fwrite(&pixel, 1, 1, file); // R
        fwrite(&pixel, 1, 1, file); // G
        fwrite(&pixel, 1, 1, file); // B
    }

    fclose(file);
}

int main() {
    int number_of_images, image_size, number_of_labels;
    uint8_t* train_images = read_mnist_images(TRAIN_IMAGES, &number_of_images, &image_size);
    unsigned char* train_labels = read_mnist_labels(TRAIN_LABELS, &number_of_labels);

    if (number_of_images != number_of_labels) {
        fprintf(stderr, "Number of images does not match number of labels\n");
        free(train_images);
        free(train_labels);
        return 1;
    }

    // Save the first few images as PPM files
    int images_to_save = 10; // Number of images to save
    for (int i = 0; i < images_to_save; i++) {
        char filename[50];
        sprintf(filename, "data/samples/mnist_image_%d_label_%d.ppm", i, train_labels[i]);
        save_ppm_image(filename, train_images + i * image_size, 28, 28);
        printf("Saved image %d with label %d as %s\n", i, train_labels[i], filename);
    }

    free(train_images);
    free(train_labels);

    return 0;
}