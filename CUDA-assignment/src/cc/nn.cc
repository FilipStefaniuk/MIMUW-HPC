#include "model.hh"
// #include "matrix.h"
#include "layers/dense.hh"
#include "layers/relu.hh"
#include "layers/sigmoid.hh"
#include "layers/softmax.hh"

#define INPUT 4096
#define LAYER_1 8192
#define LAYER_2 6144
#define LAYER_3 3072
#define LAYER_4 1024
#define OUTPUT 62

#define N 30

extern "C" {
void fit(float *data_X, float *data_Y, int len, float eps, float learning_rate, int epochs, int random) {

    // Example data
    // data_X = new float[INPUT * N];
    // data_Y = new float[OUTPUT * N];

    // for (int i = 0; i < N; ++i) {
    //     data_X[N * (i % INPUT) + i] = 1;
    //     data_Y[N * (i % OUTPUT) + i] = 1;
    // }
    //--------------

    Model model(INPUT, OUTPUT, len);
    
    model.add<Dense>(LAYER_1);
    model.add<Sigmoid>();
    model.add<Dense>(LAYER_2);
    model.add<Sigmoid>();
    model.add<Dense>(LAYER_3);
    model.add<Sigmoid>();
    model.add<Dense>(LAYER_4);
    model.add<Sigmoid>();
    model.add<Dense>(OUTPUT);
    model.add<Softmax>();

    model.summary();

    model.fit(data_X, data_Y, len, 0.00001, 0.1, 1.);

}
}