#include "relu.hh"
#include "softmax.hh"
#include "model.hh"
#include "matrix.h"

#include "stdio.h"

#define INPUT 8 //4096
#define LAYER_1 50 //8192
#define LAYER_2 20//6144
#define LAYER_3 80 //3072
#define LAYER_4 70 //1024
#define OUTPUT 4 //62

#define N 30

extern "C" {
void fit(float *data_X, float *data_Y, int len, float eps, float learning_rate, int epochs, int random) {

    // Matrix input;
    // input.height = INPUT;
    // input.width = len;
    // input.elements = data_X;

    // Matrix output;
    // output.height = OUTPUT;
    // output.width = len;
    // output.elements = data_Y;

    // matPrint(&input);

    Matrix * input = genMatrix(INPUT, N, 0);
    Matrix * output = genMatrix(OUTPUT, N, 0);

    for (int i = 0; i < input->width; ++i) {
        input->elements[input->width * (i % input->height) + i] = 1;
        output->elements[output->width * (i % output->height) + i] = 1;
    }
    // matPrint(input);
    // matPrint(output);

    // matPrintSize(input);
    // matPrintSize(output);

    ReLU layer1 = ReLU(LAYER_1);
    ReLU layer2 = ReLU(LAYER_2);
    ReLU layer3 = ReLU(LAYER_3);
    ReLU layer4 = ReLU(LAYER_4);
    Softmax layer5 = Softmax(OUTPUT);

    Model model(input, output);
    model.add(&layer1);
    model.add(&layer2);
    // model.add(&layer3);
    // model.add(&layer4);
    model.add(&layer5);

    model.build();

    model.fit(100, 0.01f, eps);
}
}