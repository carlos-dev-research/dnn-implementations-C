// Importing Standard Libraries
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Define model parameters
#define INPUT_NEURONS 784
#define OUPUT_NEURONS 10
#define ACTIVATION "softmax"

// Defin model hyperparameters
#define LEARNING_RATE 0.1
#define EPOCHS 1000

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




