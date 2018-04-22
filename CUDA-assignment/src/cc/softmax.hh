#ifndef __SOFTMAX_HH__
#define __SOFTMAX_HH__

#include "layer.hh"

class Softmax : public Layer {
    public:
        Softmax(unsigned size);

        void forward_pass(Matrix *input, Matrix *output);
        void backward_pass(Matrix* input, Matrix *g, Matrix *output);
};

#endif