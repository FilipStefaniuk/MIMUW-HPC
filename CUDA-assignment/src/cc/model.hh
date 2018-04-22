#ifndef __MODEL_HH__
#define __MODEL_HH__

#include <vector>
#include "matrix.h"
#include "layer.hh"

class Model {

    private:
        
        Matrix *input;
        Matrix *output;

        std::vector<Matrix*> gs;
        std::vector<Matrix*> gsT;
        std::vector<Matrix*> ds;
        std::vector<Layer*> layers;

        double evaluate(Matrix *pred_val, Matrix *true_val);

    public:
        Model(Matrix *input, Matrix *output);
        
        void add(Layer *layer);
        
        void build();

        void fit(unsigned epochs, float learning_rate, float eps);
};

#endif