#ifndef __MODEL_HH__
#define __MODEL_HH__

#include "../layers/layer.hh"
#include "../layers/input.hh"
#include <vector>
#include <functional>
#include <utility>

class Model {

    private:

        Matrix input, output, delta;
        int batch_size, input_size, output_size;
        std::vector<Layer*> layers;
    
    public:

        Model(int input_size, int output_size, int batch_size) 
            : input(input_size, batch_size),
              output(output_size, batch_size),
              delta(output_size, batch_size),
              batch_size(batch_size),
              input_size(input_size),
              output_size(output_size) {layers.push_back(new Input(input_size, batch_size));}

        ~Model() {
            for (Layer *layer : this->layers) {
                delete layer;
            }
        }

        template <typename L, typename... Args>
        void add(Args&& ...args) {
            layers.push_back(new L(*layers.back(), std::forward<Args>(args)...));
        }       

        float fit(float *data_x, float *data_y, int len, int epochs, float learning_rate, float eps, int random);

        void summary() {
            std::cout << std::string(40, '*') << std::endl;
            std::cout << std::string(10, ' ') << "Model Summary" << std::endl;
            std::cout << std::string(40, '*') <<std::endl;

            for (Layer *layer : this->layers) {
                std::cout << layer->info() << std::endl;
                std::cout << std::string(40, '-') << std::endl;
            }
            std::cout << std::endl;
        }

        static float crossEntropyCost(Matrix &pred_vals, Matrix &true_vals);
        
        static float accuracy(Matrix &pred_vals, Matrix &true_vals);
};

#endif