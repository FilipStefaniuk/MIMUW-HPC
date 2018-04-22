#include "layer.hh"
#include "matrix.h"

#include <stdio.h>

Layer::Layer(unsigned size) {
    this->size = size;
}

unsigned Layer::getSize() {
    return size;
}

void Layer::build(unsigned input_size) {
    dw = genMatrix(size, input_size, 0);
    wT = genMatrix(input_size, size, 0);

    w = genRandomMatrix(size, input_size);
    matTranspose(w, wT);
} 

void Layer::update_weights(Matrix* input, Matrix *delta, float eta) {
    // matPrint(input);
    // matPrint(delta);
    matMul(delta, input, dw);
    // matPrint(dw);
    matScalarMul(dw, -eta);
    // printf("DWs \n");
    // matPrint(dw);
    matSum(w, dw, w);
    // printf("Weights\n");
    // matPrint(w);
    matTranspose(w, wT);
}

Matrix* Layer::getW() {
    return w;
}

Matrix* Layer::getWT() {
    return wT;
}