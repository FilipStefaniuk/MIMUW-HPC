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
            this->weights.init();
        }

        Matrix& forward_pass(Matrix &input) {
            this->inputT = &input;
            Matrix::matMul(this->weights, input, this->output, 0);
            return this->output;
        }

        Matrix& backward_pass(Matrix &input) {
            this->d = &input;
            Matrix::matMul(weights, input, this->delta, LEFT_T);
            return this->delta;
        }

        void update(float lr) {
            Matrix::matMul(*(this->d), *(this->inputT), this->dweights, RIGHT_T);
            Matrix::matElMul(lr, this->dweights, this->dweights);
            Matrix::matSub(this->weights, this->dweights, this->weights);
        }

};

#endif