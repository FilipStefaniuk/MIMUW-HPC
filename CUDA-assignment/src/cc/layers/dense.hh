#ifndef __DENSE_HH__
#define __DENSE_HH__

#include "layer.hh"
#include <iostream>

class Dense : public Layer {
    private:

        Matrix weights, dweights;
        Matrix *d, *inputT;

    public:

        Dense(int input_size, int batch_size, int size) 
            : Layer(input_size, size, batch_size, "DENSE"),
              weights(size, input_size),
              dweights(size, input_size) {}

        Dense(Layer &prev, int size)
            : Dense(prev.getOutput().getRows(), 
              prev.getOutput().getCols(), size) {}

        ~Dense() {}

        void initialize(int flag) {
            this->weights.initialize();
        }

        Matrix& forward_pass(Matrix &input) {
            this->inputT = &input;
            Matrix::matMul(this->weights, input, this->output);
            return this->output;
        }

        Matrix& backward_pass(Matrix &input) {
            this->d = &input;
            Matrix::matMulT0(weights, input, this->delta);
            return this->delta;
        }

        void update(float lr) {
            Matrix::matMulT1(*(this->d), *(this->inputT), this->dweights);
            Matrix::matScalarMul(lr, this->dweights, this->dweights);
            Matrix::matSub(this->weights, this->dweights, this->weights);
        }

};

#endif