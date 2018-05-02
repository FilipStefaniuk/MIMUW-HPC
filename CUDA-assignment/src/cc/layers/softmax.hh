#ifndef __SOFTMAX_HH__
#define __SOFTMAX_HH__

#include "layer.hh"
#include <iostream>

class Softmax : public Layer {

    public:

        Softmax(Layer &prev) : Layer(prev, "Softmax") {}

        ~Softmax() {}

        virtual Matrix & forward_pass(Matrix &input) {

            // std::cout << "Softmax:: forward_pass:input" << std::endl;
            // std::cout << input.toString() << std::endl;
            
            Matrix::matColSoftmax(input, this->output);
            
            // std::cout << "Softmax:: forward_pass:output" << std::endl;
            // std::cout << this->output.toString() << std::endl;

            return this->output;
        }

        virtual Matrix& backward_pass(Matrix &input) {
            return input;
        }

        // virtual Matrix& build(Matrix &input);

        // virtual Matrix& forward_pass();
        
        // virtual void backward_pass(Matrix &output);

        // std::string info();
};

#endif