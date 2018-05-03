#include <cmath>
#include "softmax.hh"

#include <iostream>

//-----------------------------------------------------------------------------
//                       SOFTMAX FORWARD PASS                                       
//-----------------------------------------------------------------------------

Matrix& Softmax::forward_pass(Matrix &input) {

    // std::cout << "SOFTMAX IN" << std::endl;
    // std::cout << input.toString() << std::endl;
    // std::cout << "---------------------" << std::endl;


    for (int i = 0; i < input.getCols(); ++i) {
        
        float m = input.buff[i];

        for (int j = 0; j < input.getRows(); ++j) {
            m = fmax(m, input.buff[input.getCols() * j + i]);
        }

        float sum = 0.0f;
        for (int j = 0; j < input.getRows(); ++j) {
            sum += expf(input.buff[input.getCols() * j + i] - m);
        }

        for (int j = 0; j < input.getRows(); ++j) {
            this->output.buff[this->output.getCols() * j + i]
                = expf(input.buff[input.getCols() * j + i] - m) / sum;
        }

    }

    // std::cout << "SOFTMAX OUT" << std::endl;
    // std::cout << this->output.toString() << std::endl;
    // std::cout << "---------------------" << std::endl;

    return this->output;
}