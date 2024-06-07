// Importing Standard Libraries
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Define constants
#define INPUT_NEURONS 20
#define LEARNING_RATE 0.1
#define EPOCHS 1000

// Define customs data types
typedef struct{
    double weights[INPUT_NEURONS];
    double bias;
} NeuralNetwork;

// Initialize network
void initialize(NeuralNetwork *nn){
    for (int i = 0; i < INPUT_NEURONS; i++){
        nn->weights[i] = ((double)rand() / RAND_MAX) * 2 - 1;
    }
    nn->bias = ((double)rand() / RAND_MAX) * 2 - 1;
}

// Activation funtions
double sigmoid(double x){
    return 1 / (1 + exp(-x));
}

double sigmoid_derivative(double x){
    // x represents sigmoid(x)
    return x * (1.0 - x);
}

double forward_propagation(NeuralNetwork *nn, double *input){
    double output = 0.0;
    for (int i = 0; i < INPUT_NEURONS;i++){
        output+=input[i] * nn->weights[i];
    }
    output += nn->bias;
    output = sigmoid(output);
    return output;
}

double loss(double y_true, double y_pred){
    return pow(y_true - y_pred,2)/2.0;
}

double loss_derivative(double y_true, double y_pred ){
    return (y_true - y_pred);
}

// Binary cross-entropy loss
double binary_cross_entropy(double y_true, double y_pred) {
    return -(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred));
}

// Derivative of binary cross-entropy loss
double binary_cross_entropy_derivative(double y_true, double y_pred) {
    return (y_pred - y_true) / (y_pred * (1 - y_pred));
}

double gradient_acc(double *Y_true, double *Y_pred, int len){
    double grad = 0;
    for(int i=0;i<len;i++){
        grad+=loss_derivative(Y_true[i],Y_pred[i]);
    }
    return grad/len;
}

double Fcost(double *Y_true, double *Y_pred, int len){
    double cost = 0;
    for(int i=0;i<len;i++){
        cost+=loss(Y_true[i],Y_pred[i]);
    }
    return cost/len;
}

double Fcost_binary(double *Y_true, double *Y_pred, int len){
    double cost = 0;
    for(int i=0;i<len;i++){
        cost+=binary_cross_entropy(Y_true[i],Y_pred[i]);
    }
    return cost/len;
}

void backpropagation(NeuralNetwork *nn, double *input, double target, double* grad_W,double* grad_B){
    // Calculate forward pass and stored results in output array previously allocated
    double output = forward_propagation(nn,input);

    // Start Backpropagation
    //double grad_dL=loss_derivative(target,output);
    double grad_dL=binary_cross_entropy_derivative(target,output);
    double grad_dout = output * (1.0 - output);
    double* grad_zW = input;
    double grad_zB = 1;
    for (int i=0;i<INPUT_NEURONS;i++){
        grad_W[i] += LEARNING_RATE*grad_dL*grad_dout*grad_zW[i];
    }
    *grad_B += LEARNING_RATE*grad_dL*grad_dout*grad_zB;
}

int main(){
    // Create network object
    NeuralNetwork nn;

    // Initialze parameters random if not available
    initialize(&nn);

    // Load input data (AND logic gate)
    double input[4][INPUT_NEURONS] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    double target[] = {0, 0, 0, 1};

    int n_samples = 4; // Hard code for example
    // Training
    for (int epoch=0;epoch<EPOCHS;epoch++){
        printf("Starting Epoch %d\n",epoch);
        double grad_W[INPUT_NEURONS];
        for (int i=0;i<INPUT_NEURONS;i++){
            grad_W[i]=0;
        }
        double grad_B=0;

        for (int i=0;i<n_samples;i++){
            backpropagation(&nn,input[i],target[i],grad_W,&grad_B);
        }

        //Update parameters
        for (int i=0;i<INPUT_NEURONS;i++){
            nn.weights[i]-=grad_W[i];
        }
        nn.bias-=grad_B;

        // Get Cost function to estimate progress
        double cost = 0;
        for (int i=0;i<n_samples;i++){
            double output = forward_propagation(&nn,input[i]);
            cost+=binary_cross_entropy(target[i],output);
        }
        printf("Epoch %d, Cost=%.2f",epoch,cost);
    }

    // Save Weights if neccessary
    // Print final results
    for (int i = 0; i < n_samples; i++) {
        double output = forward_propagation(&nn, input[i]);
        printf("\nInput: [%d, %d] - Predicted: %.5f - Actual: %.1f\n", (int)input[i][0], (int)input[i][1], output, target[i]);
    }

    return 0;


}