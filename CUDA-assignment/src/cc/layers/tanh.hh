#ifndef __TANH_HH__
#define __TANH_HH__

#include "layer.hh"

class Tanh : public Layer {
    
    private:
        Matrix outputPrime;

    public:

        Tanh(Layer &prev) 
        : Layer(prev, "Sigmoid"),  outputPrime(prev.getOutput().getRows(), prev.getOutput().getCols()) {}

        ~Tanh() {}

        virtual Matrix& forward_pass(Matrix &input) {
            Matrix::matTanh(input, this->output);
            Matrix::matTanhPrime(input, this->outputPrime);

            // std::cout << this->output.toString() << std::endl;

            return this->output;
        }

        virtual Matrix& backward_pass(Matrix &input) {
            Matrix::matElMul(input, this->outputPrime, this->delta);

            // std::cout << this->delta.toString() << std::endl;

            return this->delta;
        }

        // int build(int data_len, int prev_size);

        // virtual Matrix& forward_pass();
        
        // virtual void backward_pass(Matrix &output);

        // std::string info();

};

#endif