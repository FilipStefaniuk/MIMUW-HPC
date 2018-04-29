#include "softmax.hh"

#ifndef NDEBUG
    const bool debug = true;
#else
    const bool debug = false;
#endif

Matrix& Softmax::build(Matrix &input) {
    // g = genMatrix(input.height, input.width, 0);
    // d = genMatrix(input.height, input.width, 0);
    // return Layer::build(*g);
}

Matrix& Softmax::forward_pass() {
    // if (this->prev != this) {
    //     matSoftmax(&(this->prev->getg()), g);
    // }
    // return Layer::forward_pass();
}

void Softmax::backward_pass(Matrix &output) {}

std::string Softmax::info() {
    return "Softmax";
}