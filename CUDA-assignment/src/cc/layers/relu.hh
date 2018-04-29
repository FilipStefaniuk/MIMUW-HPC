#ifndef __RELU_HH__
#define __RELU_HH__

#include "layer.hh"

class ReLU : public Layer {
    
    private:
        Matrix outputPrime;

    public:

        ReLU(Layer &prev) 
        : Layer(prev, "ReLU"),  outputPrime(prev.getOutput().getRows(), prev.getOutput().getCols()) {}

        ~ReLU() {}

        virtual Matrix& forward_pass(Matrix &input) {
            Matrix::matReLU(input, this->output);
            Matrix::matReLUPrime(input, this->outputPrime);

            // std::cout << "ReLU::forward_pass: output" << std::endl;
            // std::cout << this->output.toString() << std::endl;

            return this->output;
        }

        virtual Matrix& backward_pass(Matrix &input) {
            Matrix::matElMul(input, this->outputPrime, this->delta);

            // std::cout << "ReLU::backward_pass: output" << std::endl;
            // std::cout << this->delta.toString() << std::endl;

            return this->delta;
        }

        // int build(int data_len, int prev_size);

        // virtual Matrix& forward_pass();
        
        // virtual void backward_pass(Matrix &output);

        // std::string info();

};

#endif