#include "model/model.hh"
#include "layers/dense.hh"
#include "layers/tanh.hh"
#include "layers/relu.hh"
#include "layers/softmax.hh"
#include <iostream>

#define INPUT 4096
#define LAYER_1 8192
#define LAYER_2 6144
#define LAYER_3 3072
#define LAYER_4 1024
#define OUTPUT 62

#define BATCH_SIZE 32

extern "C" {

float fit(float *data_X, float *data_Y, int len, float eps, float learning_rate, int epochs, int random) {

    // for (int i = 0; i < INPUT; ++i) {
    //     for (int j = 0; j < 10; ++j) {
    //         if (j)
    //             std::cout << " ";
    //         std::cout << data_X[i * len + j];
    //     }
    //     std::cout << std::endl;
    // }

    Model model(INPUT, OUTPUT, BATCH_SIZE);
    
    model.add<Dense>(LAYER_1);
    model.add<ReLU>();
    model.add<Dense>(LAYER_2);
    model.add<ReLU>();
    model.add<Dense>(LAYER_3);
    model.add<ReLU>();
    model.add<Dense>(LAYER_4);
    model.add<ReLU>();
    model.add<Dense>(OUTPUT);
    model.add<Softmax>();

    model.summary();

    return model.fit(data_X, data_Y, len, epochs, learning_rate, eps, random);
}
}