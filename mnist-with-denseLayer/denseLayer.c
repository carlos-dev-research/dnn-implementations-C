// Importing Standard Libraries
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

// Define model parameters
#define INPUT_NEURONS 784
#define OUTPUT_NEURONS 10
#define ACTIVATION "softmax"
#define EPSILON 1e-9


// Defin model hyperparameters
#define LEARNING_RATE 0.01
#define EPOCHS 2

// Dataset paths
#define TRAIN_IMAGES "data/train-images.idx3-ubyte"
#define TRAIN_LABELS "data/train-labels.idx1-ubyte"
#define TEST_IMAGES "data/t10k-images.idx3-ubyte"
#define TEST_LABELS "data/t10k-labels.idx1-ubyte"

// Define customs data types
typedef struct{
    float weights[OUTPUT_NEURONS][INPUT_NEURONS];
    float bias[OUTPUT_NEURONS];
} NeuralNetwork;

// Initialize network
void initialize(NeuralNetwork *nn){
    for (int i = 0; i < OUTPUT_NEURONS; i++){
        for (int j=0; j < INPUT_NEURONS; j++){
            nn->weights[i][j] = ((float)rand() / RAND_MAX) * 2 - 1;
        }
         nn->bias[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }
    // Debug: Print initial weights and biases
    printf("Initial weights and biases:\n");
    for (int i = 0; i < OUTPUT_NEURONS; i++) {
        printf("Neuron %d: Bias = %.5f\n", i, nn->bias[i]);
        for (int j = 0; j < INPUT_NEURONS; j++) {
            if (j < 5) {  // Only print the first 5 weights for brevity
                printf("Weight[%d][%d] = %.5f ", i, j, nn->weights[i][j]);
            }
        }
        printf("\n");
    }
}

// Function to normalize the pixel values
void process_images(uint8_t *image_data, float **normalized_images, int num_images, int image_size) {
    for (int i = 0; i < num_images; i++) {
        for (int j = 0; j < image_size; j++) {
            normalized_images[i][j] = image_data[i * image_size + j] / 255.0f;
        }
    }
}


// Loss function and its derivative
// Function to calculate the softmax
void softmax(float *input, float *output, int length) {
    float max_input = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_input) {
            max_input = input[i];
        }
    }
    

    float sum = 0.0;
    for (int i = 0; i < length; i++) {
        output[i] = exp(input[i]-max_input);
        sum += output[i];
    }

    for (int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}

// Function to calculate the gradient of the cross-entropy loss with softmax
void softmax_cross_entropy_loss_derivative(float *softmax_output, float *target, float *derivative, int length) {
    for (int i = 0; i < length; i++) {
        derivative[i] = softmax_output[i] - target[i];
    }
}


// Function to calculate cross-entropy loss
float cross_entropy_loss(float *softmax_output, float *target, int length) {
    float loss = 0.0;
    for (int i = 0; i < length; i++) {
        float output_clamped = fmaxf(softmax_output[i], EPSILON); // Prevent log(0)
        loss -= target[i] * log(output_clamped);
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

void save_images(int images_to_save,uint8_t* train_images,unsigned char* train_labels,int image_size){
        for (int i = 0; i < images_to_save; i++) {
            char filename[50];
            sprintf(filename, "data/samples/mnist_image_%d_label_%d.ppm", i, train_labels[i]);
            save_ppm_image(filename, train_images + i * image_size, 28, 28);
            printf("Saved image %d with label %d as %s\n", i, train_labels[i], filename);
        }
}

// Forward propagation
void forward_prop(NeuralNetwork *nn, float* pixels, float* classes) {
    float z[OUTPUT_NEURONS] = {0};

    for (int i = 0; i < OUTPUT_NEURONS; i++) {
        for (int j = 0; j < INPUT_NEURONS; j++) {
            z[i] += pixels[j] * nn->weights[i][j];
        }
        z[i] += nn->bias[i];
    }
    softmax(z, classes, OUTPUT_NEURONS);
}

// Backward propagation
void back_prop(NeuralNetwork *nn, float* pixels, float* output, float* target) {
    float dz[OUTPUT_NEURONS] = {0};
    softmax_cross_entropy_loss_derivative(output, target, dz, OUTPUT_NEURONS);

    for (int i = 0; i < OUTPUT_NEURONS; i++) {
        float delta = dz[i] * output[i] * (1 - output[i]);  // Dactivationj/Dzj
        for (int j = 0; j < INPUT_NEURONS; j++) {
            nn->weights[i][j] -= LEARNING_RATE * delta * pixels[j];  // Dzj/Dwij
        }
        nn->bias[i] -= LEARNING_RATE * delta;  // Dzj/Dbj
    }
}

// Function to print final results
void print_final_results(NeuralNetwork *nn, float **images, unsigned char *labels, int num_samples) {
    float output[OUTPUT_NEURONS] = {0};
    for (int i = 0; i < num_samples; i++) {
        forward_prop(nn, images[i], output);

        // Find the predicted class
        int predicted_class = 0;
        float max_value = output[0];
        for (int j = 1; j < OUTPUT_NEURONS; j++) {
            if (output[j] > max_value) {
                max_value = output[j];
                predicted_class = j;
            }
        }

        printf("Sample %d - Predicted: %d - Actual: %d\n", i, predicted_class, labels[i]);
    }
}


int main() {
    time_t t;
    srand((unsigned) time(&t));  // Seed the random number generator
    printf("Seed: %ld\n", t);  // Print the seed value for debugging
    
    int number_of_images, image_size, number_of_labels;
    uint8_t* train_images = read_mnist_images(TRAIN_IMAGES, &number_of_images, &image_size);
    unsigned char* train_labels = read_mnist_labels(TRAIN_LABELS, &number_of_labels);
    NeuralNetwork nn;
    initialize(&nn);

    if (number_of_images != number_of_labels) {
        fprintf(stderr, "Number of images does not match number of labels\n");
        free(train_images);
        free(train_labels);
        return 1;
    }

    
    // Normalize and separate images
    float **normalized_images = (float**)malloc(number_of_images * sizeof(float*));
    for (int i = 0; i < number_of_images; i++) {
        normalized_images[i] = (float*)malloc(image_size * sizeof(float));
    }
    process_images(train_images, normalized_images, number_of_images, image_size);

    // Convert labels to one-hot encoding
    float **one_hot_labels = (float **)malloc(number_of_labels * sizeof(float *));
    for (int i = 0; i < number_of_labels; ++i) {
        one_hot_labels[i] = (float *)calloc(OUTPUT_NEURONS, sizeof(float));
    }
    for (int i = 0; i < number_of_labels; ++i) {
        one_hot_labels[i][train_labels[i]] = 1.0f;
    }

    // Training loop
    float output[OUTPUT_NEURONS] = {0};
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        time_t start_time = time(NULL);
        for (int i = 0; i < number_of_images; i++) {
            forward_prop(&nn, normalized_images[i], output);
            back_prop(&nn, normalized_images[i], output, one_hot_labels[i]);
        }
        // Debugging: Print loss every 100 epochs
        
        float total_loss = 0.0;
        for (int i = 0; i < number_of_images; i++) {
            forward_prop(&nn, normalized_images[i], output);
            total_loss += cross_entropy_loss(output, one_hot_labels[i], OUTPUT_NEURONS);
        }
        time_t end_time = time(NULL);
        double epoch_time = difftime(end_time, start_time);
        printf("Epoch %d, Loss: %.4f,Time: %.4f seconds\n", epoch, total_loss / number_of_images,epoch_time);
        
    }

    // Print final results for the first 10 samples
    print_final_results(&nn, normalized_images, train_labels, 10);

    // Free allocated memory
    for (int i = 0; i < number_of_images; i++) {
        free(normalized_images[i]);
    }
    free(normalized_images);

    for (int i = 0; i < number_of_labels; ++i) {
        free(one_hot_labels[i]);
    }
    free(one_hot_labels);

    free(train_images);
    free(train_labels);

    return 0;
}