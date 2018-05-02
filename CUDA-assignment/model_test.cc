#include "src/cc/model.hh"
#include "src/cc/layers/dense.hh"
#include "src/cc/layers/relu.hh"
#include "src/cc/layers/tanh.hh"
#include "src/cc/layers/softmax.hh"
// #include "src/cc/layers/sigmoid.hh"
#include "src/cc/initializers/random_initializer.hh"
#include "src/cc/initializers/buffer_initializer.hh"

#define LEN 128
#define BATCH_SIZE 128 //4608 // 4591// 32

#define INPUT 4096
#define LAYER_1 128 //8192
#define LAYER_2 6144
#define LAYER_3 3072
#define LAYER_4 1024
#define OUTPUT 62

int main() {

    
    float *data_x = new float[INPUT * LEN];
    float *data_y = new float[OUTPUT * LEN];
    // BufferInitializer initX(data_x), initY(data_y);
    
    for (int i = 0; i < LEN; ++i) {
        data_x[LEN * (i % INPUT) + i] = 1;
        data_y[LEN * (i % OUTPUT) + i] = 1;
    }

    // initX.fill(data_x, INPUT * BATCH_SIZE);
    // initY.fill(data_y, OUTPUT * BATCH_SIZE);

    Model model(INPUT, OUTPUT, BATCH_SIZE);
    
    model.add<Dense>(LAYER_1);
    model.add<Tanh>();
    // model.add<Dense>(LAYER_2);
    // model.add<Tanh>();
    // model.add<Dense>(LAYER_3);
    // model.add<Tanh>();
    // model.add<Dense>(LAYER_4);
    // model.add<Tanh>();
    model.add<Dense>(OUTPUT);
    model.add<Softmax>();

    model.summary();

    model.fit(data_x, data_y, LEN, 100, 0.01, 0., 1.);

    delete data_x;
    delete data_y;
}