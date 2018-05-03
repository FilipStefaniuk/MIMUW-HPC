#include "model.hh"
#include "layers/dense.hh"
#include "layers/tanh.hh"
#include "layers/relu.hh"
#include "layers/softmax.hh"
#include <iostream>

#define MNIST_INPUT 784
#define MNIST_LAYER_1 30
#define MNIST_OUTPUT 10

// #define INPUT 4096
// #define LAYER_1 100 //8192
// #define LAYER_2 512 //6144
// #define LAYER_3 3072
// #define LAYER_4 1024
// #define OUTPUT 62

extern "C" {
void fitMNIST(float *data_X, float *data_Y, int len, float eps, float lr, int epochs, int random) {
    Model model(MNIST_INPUT, MNIST_OUTPUT, len);

    model.add<Dense>(MNIST_LAYER_1);
    model.add<Tanh>();
    model.add<Dense>(MNIST_OUTPUT);
    model.add<Softmax>();

    model.summary();

    model.fit(data_X, data_Y, len, epochs, lr, eps, random);
}


// void fit(float *data_X, float *data_Y, int len, float eps, float learning_rate, int epochs, int random) {

//     Model model(INPUT, OUTPUT, len);
    
//     model.add<Dense>(LAYER_1);
//     model.add<Tanh>();
//     model.add<Dense>(LAYER_2);
//     model.add<Tanh>();
//     model.add<Dense>(LAYER_3);
//     model.add<Tanh>();
//     model.add<Dense>(LAYER_4);
//     model.add<Tanh>();
//     model.add<Dense>(OUTPUT);
//     model.add<Softmax>();

//     model.summary();

//     model.fit(data_X, data_Y, len, epochs, learning_rate, eps, random);

// }
}