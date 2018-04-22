#include "relu.hh"
#include <stdio.h>
#include "matrix.h"

ReLU::ReLU(unsigned size) : Layer(size) {}

void ReLU::forward_pass(Matrix *input, Matrix *output) {
    // matPrintSize(getW());
    // matPrintSize(input);
    // matPrintSize(output);
    matMul(getW(), input, output);
    matRelu(output);
    // printf("output\n");
    // matPrint(output);
}

void ReLU::backward_pass(Matrix *input, Matrix *g, Matrix *output) {
    
    printf("ReLU Layer Back\n");
    matReluBack(g);
    // matPrint(g);
    matElMul(input, g, input);
    // matPrint(input);
    // matPrint(getWT());
    matMul(getWT(), input, output);
    // printf("ReLU delta\n");
    // matPrint(output);
}