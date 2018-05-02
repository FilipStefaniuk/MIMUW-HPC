#include "model.hh"
#include "layers/dense.hh"
#include "layers/tanh.hh"
#include "layers/relu.hh"
#include "layers/softmax.hh"

#define INPUT 4096
#define LAYER_1 8192
#define LAYER_2 6144
#define LAYER_3 3072
#define LAYER_4 1024
#define OUTPUT 62

extern "C" {
void fit(float *data_X, float *data_Y, int len, float eps, float learning_rate, int epochs, int random) {

    Model model(INPUT, OUTPUT, len);
    
    model.add<Dense>(LAYER_1);
    model.add<Tanh>();
    model.add<Dense>(LAYER_2);
    model.add<Tanh>();
    model.add<Dense>(LAYER_3);
    model.add<Tanh>();
    model.add<Dense>(LAYER_4);
    model.add<Tanh>();
    model.add<Dense>(OUTPUT);
    model.add<Softmax>();

    model.summary();

    model.fit(data_X, data_Y, len, epochs, 0.00001, 0.1, 1.);

}
}