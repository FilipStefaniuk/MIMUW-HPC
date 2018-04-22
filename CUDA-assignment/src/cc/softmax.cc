#include "softmax.hh"
#include "matrix.h"
#include <stdio.h>

Softmax::Softmax(unsigned size) : Layer(size) {}

void Softmax::forward_pass(Matrix *input, Matrix *output) {
    matMul(getW(), input, output);
    matSoftmax(output);
}

void Softmax::backward_pass(Matrix *input, Matrix *g, Matrix *output) {

    // printf("SOFTMAX INPUT:");
    // matPrint(input);
    // printf("SOFTMAX WEIGHTS TRANSPOSED");
    // matPrint(getWT());
    matMul(getWT(), input, output);
    
    // printf("Softmax delta:\n");
    // matPrint(output);

}