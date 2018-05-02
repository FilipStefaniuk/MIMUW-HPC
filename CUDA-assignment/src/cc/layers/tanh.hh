#ifndef __TANH_HH__
#define __TANH_HH__

#include "layer.hh"

class Tanh : public Layer {
    
    private:

        Matrix outputPrime;

        void activation(Matrix &input, Matrix &output, Matrix &output_prime);

    public:

        Tanh(Layer &prev) 
        : Layer(prev, "Tanh"),
          outputPrime(prev.getOutput().getRows(), 
          prev.getOutput().getCols()) {}

        ~Tanh() {}

        virtual Matrix& forward_pass(Matrix &input) {
            activation(input, this->output, this->outputPrime);
            return this->output;
        }

        virtual Matrix& backward_pass(Matrix &input) {
            Matrix::matElMul(input, this->outputPrime, this->delta);
            return this->delta;
        }
};

#endif