#ifndef __MODEL_HH__
#define __MODEL_HH__

#include "layers/layer.hh"
#include <vector>
#include <functional>
#include <utility>

class Model {

    private:

        Matrix input, output, delta;
        std::vector<Layer*> layers;
    
    public:

        Model(int input_size, int output_size, int batch_size);
        ~Model();

        template <typename L, typename... Args>
        void add(Args&& ...args) {
            layers.push_back(new L(*layers.back(), std::forward<Args>(args)...));
        }       

        void fit(float *data_x, float *data_y, int epochs, float learning_rate, float eps, int random);
        void summary();
};

#endif