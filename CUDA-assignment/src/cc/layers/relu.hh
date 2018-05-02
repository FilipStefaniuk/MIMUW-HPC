#ifndef __RELU_HH__
#define __RELU_HH__

#include "layer.hh"

class ReLU : public Layer {
    
    private:
        Matrix outputPrime;

        void activation(Matrix &input, Matrix &output, Matrix &output_prime);

    public:

        ReLU(Layer &prev) 
        : Layer(prev, "ReLU"),  
          outputPrime(prev.getOutput().getRows(), 
          prev.getOutput().getCols()) {}

        ~ReLU() {}

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