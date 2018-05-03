#include "relu.hh"

//-----------------------------------------------------------------------------
//                            RELU ACTIVATION                                       
//-----------------------------------------------------------------------------

void ReLU::activation(Matrix &input, Matrix &output, Matrix &output_prime) {
    
    for (int i = 0; i < input.getRows(); ++i) {
        
        for (int j = 0; j < input.getCols(); ++j) {
        
            float tmp = input.buff[i * input.getCols() + j];
            output.buff[i * output.getCols() + j] = tmp > 0 ? tmp : 0;
            output_prime.buff[i * output_prime.getCols() + j] = tmp > 0 ? 1 : 0;
        }
    }
}
