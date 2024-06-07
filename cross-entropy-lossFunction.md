# Derivation of Cross-Entropy Loss Function Gradient with Softmax

## 1. Softmax Function

The softmax function for a vector \( \mathbf{z} \) with elements \( z_i \) is defined as:

\[ y_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}} \]

where \( y_i \) is the output of the softmax function for the \( i \)-th element.

## 2. Cross-Entropy Loss Function

The cross-entropy loss function for a single training example is defined as:

\[ L = - \sum_{i} t_i \log(y_i) \]

where \( t_i \) is the target probability for class \( i \), and \( y_i \) is the predicted probability for class \( i \) from the softmax function.

## 3. Combined Softmax and Cross-Entropy Loss

To derive the gradient of the loss with respect to the input \( \mathbf{z} \), we need to calculate \( \frac{\partial L}{\partial z_i} \).

## 4. Derivation Steps

**Step 1: Express the Loss**

\[ L = - \sum_{i} t_i \log\left( \frac{e^{z_i}}{\sum_{j} e^{z_j}} \right) \]

**Step 2: Simplify the Logarithm**

Using properties of logarithms:

\[ L = - \sum_{i} t_i \left( \log(e^{z_i}) - \log\left( \sum_{j} e^{z_j} \right) \right) \]

\[ L = - \sum_{i} t_i \left( z_i - \log\left( \sum_{j} e^{z_j} \right) \right) \]

**Step 3: Separate the Sum**

\[ L = - \sum_{i} t_i z_i + \sum_{i} t_i \log\left( \sum_{j} e^{z_j} \right) \]

\[ L = - \sum_{i} t_i z_i + \log\left( \sum_{j} e^{z_j} \right) \sum_{i} t_i \]

Since \( \sum_{i} t_i = 1 \) (target probabilities sum to 1):

\[ L = - \sum_{i} t_i z_i + \log\left( \sum_{j} e^{z_j} \right) \]

**Step 4: Derivative of the Loss Function**

To find \( \frac{\partial L}{\partial z_i} \), we need to differentiate both terms in the expression for \( L \).

\[ \frac{\partial L}{\partial z_i} = \frac{\partial}{\partial z_i} \left( - \sum_{k} t_k z_k + \log\left( \sum_{j} e^{z_j} \right) \right) \]

The first term is straightforward:

\[ \frac{\partial}{\partial z_i} \left( - \sum_{k} t_k z_k \right) = - t_i \]

The second term is:

\[ \frac{\partial}{\partial z_i} \log\left( \sum_{j} e^{z_j} \right) = \frac{1}{\sum_{j} e^{z_j}} \cdot e^{z_i} = \frac{e^{z_i}}{\sum_{j} e^{z_j}} = y_i \]

Therefore:

\[ \frac{\partial L}{\partial z_i} = y_i - t_i \]

## Summary

The gradient of the cross-entropy loss with respect to the input \( \mathbf{z} \) when using the softmax function is:

\[ \frac{\partial L}{\partial z_i} = y_i - t_i \]

This result is very convenient because it simplifies backpropagation. For each class \( i \), the gradient is simply the difference between the predicted probability \( y_i \) and the target probability \( t_i \).

## Implementation in C

This theoretical result translates directly into the C implementation we discussed earlier, where we compute the gradient as:

```c
void softmax_cross_entropy_loss_derivative(double *softmax_output, int *target, double *derivative, int length) {
    for (int i = 0; i < length; i++) {
        derivative[i] = softmax_output[i] - target[i];
    }
}
