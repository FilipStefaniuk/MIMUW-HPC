#include <cmath>
#include "tanh.hh"

//-----------------------------------------------------------------------------
//                            TANH ACTIVATION                                       
//-----------------------------------------------------------------------------

void Tanh::activation(Matrix &input, Matrix &output, Matrix &output_prime) {

    // std::cout << "TANH in" << std::endl;
    // std::cout << input.toString() << std::endl;
    // std::cout << "---------------------" << std::endl;

    for (int i = 0; i < input.getRows(); ++i) {
        
        for (int j = 0; j < input.getCols(); ++j) {
        
            float tmp = tanh(input.buff[i * input.getCols() + j]);
            output.buff[i * output.getCols() + j] = tmp;
            output_prime.buff[i * output_prime.getCols() + j] = 1 - tmp;
        }
    }

    // std::cout << "TANH out" << std::endl;
    // std::cout << output.toString() << std::endl;
    // std::cout << "---------------------" << std::endl;

}