#ifndef __INPUT_HH__
#define __INPUT_HH__

#include "layer.hh"
#include <iostream>

class Input : public Layer {

    public:
        Input(int input_size, int batch_size) 
            : Layer(input_size, input_size, batch_size, "INPUT") {} 

        ~Input() {};

        virtual void initialize(Initializer &initializer) {
            // this->output.initialize(initializer);
        }

        virtual Matrix & forward_pass(Matrix &input) {
            // std::cout << input.toString() << std::endl;
            return input;
        }

        virtual Matrix& backward_pass(Matrix &input) {
            return input;
        }

        // Input(Layer &prev) 
            // : Layer(1, 1) {this->name = "INPUT";}
        

        // virtual Matrix& build(Matrix &input);

        // virtual Matrix& forward_pass();
        
        // virtual void backward_pass(Matrix &output);

        // unsigned getSize();

};

#endif