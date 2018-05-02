#ifndef __SOFTMAX_HH__
#define __SOFTMAX_HH__

#include "layer.hh"
#include <iostream>

class Softmax : public Layer {

    public:

        Softmax(Layer &prev) : Layer(prev, "Softmax") {}

        ~Softmax() {}

        virtual Matrix & forward_pass(Matrix &input);

        virtual Matrix& backward_pass(Matrix &input) {
            return input;
        }
};

#endif