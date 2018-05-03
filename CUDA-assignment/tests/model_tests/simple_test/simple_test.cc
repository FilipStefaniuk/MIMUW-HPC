#include "../../../src/cc/model/model.hh"
#include "../../../src/cc/layers/dense.hh"
#include "../../../src/cc/layers/relu.hh"
#include "../../../src/cc/layers/tanh.hh"
#include "../../../src/cc/layers/softmax.hh"
#include <iostream>
#include <sstream>
#include <string>

std::string buffToString(float *data, int rows, int cols) {
    std::stringstream ss;
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (j) {
                ss << " ";
            }
            ss << data[cols * i + j];
        }
        ss << "\n";
    }

    return ss.str();
}

void test_model(Model &model, int input, int output, int epochs, float lr) {

    float *data_x = new float[input * input];
    float *data_y = new float[output * input];
    
    for (int i = 0; i < input; ++i) {
        data_x[input * (i % input) + i] = 1;
        data_y[input * (i % output) + i] = 1;
    }

    // std::cout << buffToString(data_x, input, input) << std::endl;
    // std::cout << buffToString(data_y, output, input) << std::endl;

    model.fit(data_x, data_y, input, epochs, lr, 0., 1.);

    delete data_x;
    delete data_y;
}

int main() {

    {
        std::cout << "DIVISIBILITY BY 3 (SMALL NN)" << std::endl;
        std::cout << "---------------------" << std::endl;

        int const input = 20;
        int const output = 3;
        
        Model model(input, output, input);
    
        model.add<Dense>(5);
        model.add<ReLU>();
        model.add<Dense>(output);
        model.add<Softmax>();

        test_model(model, input, output, 20, 1.4);
        std::cout << std::endl;
    }

    {
        std::cout << "DIVISIBILITY BY 7" << std::endl;
        std::cout << "---------------------" << std::endl;

        int const input = 150;
        int const output = 7;
        
        Model model(input, output, input);
    
        model.add<Dense>(20);
        model.add<ReLU>();
        model.add<Dense>(output);
        model.add<Softmax>();

        test_model(model, input, output, 100, 0.7);
        std::cout << std::endl;
    }

    {
        std::cout << "DIVISIBILITY BY 2 (DEEP NN) " << std::endl;
        std::cout << "---------------------" << std::endl;

        int const input = 5;
        int const output = 2;
        
        Model model(input, output, input);

        model.add<Dense>(2);
        model.add<ReLU>();    
        model.add<Dense>(2);
        model.add<ReLU>();    
        model.add<Dense>(2);
        model.add<ReLU>();
        model.add<Dense>(output);
        model.add<Softmax>();

        test_model(model, input, output, 100, 0.1);
        std::cout << std::endl;
    }
}