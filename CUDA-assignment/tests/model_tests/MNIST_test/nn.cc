#include "../../../src/cc/model/model.hh"
#include "../../../src/cc/layers/dense.hh"
#include "../../../src/cc/layers/tanh.hh"
#include "../../../src/cc/layers/relu.hh"
#include "../../../src/cc/layers/softmax.hh"
#include <iostream>

#define MNIST_INPUT 784
#define MNIST_LAYER_1 100
#define MNIST_LAYER_2 30
#define MNIST_OUTPUT 10

#define BATCH_SIZE 32

extern "C" {
void fitMNIST(float *data_X, float *data_Y, int len, float eps, float lr, int epochs, int random) {
    
    // for (int i =0; i < MNIST_INPUT; ++i) {
    //     for (int j = 0; j < len; ++j) {
    //         if (j)
    //             std::cout << " ";
    //         std::cout << data_X[i * len + j];
    //     }
    //     std::cout << std::endl;
    // }

    Model model(MNIST_INPUT, MNIST_OUTPUT, BATCH_SIZE);

    // model.add<Dense>(MNIST_LAYER_1);
    // model.add<ReLU>();
    model.add<Dense>(MNIST_LAYER_2);
    model.add<ReLU>();
    model.add<Dense>(MNIST_OUTPUT);
    model.add<Softmax>();

    model.summary();

    model.fit(data_X, data_Y, len, epochs, lr, eps, random);
}
}