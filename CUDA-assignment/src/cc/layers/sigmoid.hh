#ifndef __SIGMOID_HH__
#define __SIGMOID_HH__

#include "layer.hh"

class Sigmoid : public Layer {
    
    private:
        Matrix outputPrime;

    public:

        Sigmoid(Layer &prev) 
        : Layer(prev, "Sigmoid"),  outputPrime(prev.getOutput().getRows(), prev.getOutput().getCols()) {}

        ~Sigmoid() {}

        virtual Matrix& forward_pass(Matrix &input) {
            Matrix::matSigmoid(input, this->output);
            Matrix::matSigmoidPrime(input, this->outputPrime);

            // std::cout << "Sigmoid::forward_pass: output:" << std::endl;
            // std::cout << this->output.toString() << std::endl;

            return this->output;
        }

        virtual Matrix& backward_pass(Matrix &input) {
            Matrix::matElMul(input, this->outputPrime, this->delta);

            // std::cout << "Sigmoid::backward_pass: delta:" << std::endl;
            // std::cout << this->delta.toString() << std::endl;

            return this->delta;
        }

        // int build(int data_len, int prev_size);

        // virtual Matrix& forward_pass();
        
        // virtual void backward_pass(Matrix &output);

        // std::string info();

};

#endif