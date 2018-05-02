#include "src/cc/model.hh"
#include "src/cc/layers/dense.hh"
#include "src/cc/layers/relu.hh"
#include "src/cc/layers/tanh.hh"
#include "src/cc/layers/softmax.hh"

#define LEN 4591

#define INPUT 4096
#define LAYER_1 8192
#define LAYER_2 6144
#define LAYER_3 3072
#define LAYER_4 1024
#define OUTPUT 62

int main() {

    
    float *data_x = new float[INPUT * LEN];
    float *data_y = new float[OUTPUT * LEN];
    
    for (int i = 0; i < LEN; ++i) {
        data_x[LEN * (i % INPUT) + i] = 1;
        data_y[LEN * (i % OUTPUT) + i] = 1;
    }

    Model model(INPUT, OUTPUT, LEN);
    
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

    model.fit(data_x, data_y, LEN, 10, 0.000000001, 0., 1.);

    delete data_x;
    delete data_y;
}