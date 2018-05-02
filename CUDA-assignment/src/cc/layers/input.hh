#ifndef __INPUT_HH__
#define __INPUT_HH__

#include "layer.hh"
#include <iostream>

class Input : public Layer {

    public:
        Input(int input_size, int batch_size) 
            : Layer(input_size, input_size, batch_size, "INPUT") {} 

        ~Input() {};

        virtual Matrix & forward_pass(Matrix &input) {
            return input;
        }

        virtual Matrix& backward_pass(Matrix &input) {
            return input;
        }
};

#endif