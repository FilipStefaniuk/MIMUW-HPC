#ifndef __RELU_HH__
#define __RELU_HH__

#include "layer.hh"

class ReLU : public Layer {
    public:
        ReLU(unsigned size);

        void forward_pass(Matrix *input, Matrix *output);
        void backward_pass(Matrix* input, Matrix *g, Matrix *output);

};

#endif