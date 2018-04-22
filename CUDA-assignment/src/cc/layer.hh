#ifndef __LAYER_H__
#define __LAYER_H__

#include "matrix.h"

class Layer {
    
    private:
    
        unsigned size;
        
        Matrix *w;
        Matrix *wT;
        Matrix *dw;

    public:
        Layer(unsigned size);

        void build(unsigned input_size);

        void update_weights(Matrix* input, Matrix *delta, float eta);

        virtual void forward_pass(Matrix *input, Matrix *output) = 0;
        
        virtual void backward_pass(Matrix* input, Matrix *g, Matrix *output) = 0; 

        unsigned getSize();

        Matrix* getW();

        Matrix* getWT();
};

#endif