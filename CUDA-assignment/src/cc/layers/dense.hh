#ifndef __DENSE_HH__
#define __DENSE_HH__

#include "layer.hh"
#include <iostream>

class Dense : public Layer {
    private:

        Matrix *inputT;
        Matrix weights, /*weightsT,*/ dweights;
        Matrix *d;

    public:

        Dense(int input_size, int batch_size, int size) 
            : Layer(input_size, size, batch_size, "DENSE"),
            //   inputT(batch_size, input_size), 
              weights(size, input_size),
            //   weightsT(input_size, size),
              dweights(size, input_size) {}

        Dense(Layer &prev, int size)
            : Dense(prev.getOutput().getRows(), prev.getOutput().getCols(), size) {}

        ~Dense() {}

        virtual void initialize(Initializer &initializer) {
            this->weights.initialize();

            // Matrix::matT(this->weights, this->weightsT);
        }

        virtual Matrix& forward_pass(Matrix &input) {
            // Matrix::matT(input, this->inputT);
            this->inputT = &input;
            Matrix::matMul(this->weights, input, this->output);
            
            // std::cout << "Dense::forward_pass: output" << std::endl;
            // std::cout << this->output.toString() << std::endl;
            
            return this->output;
        }

        virtual Matrix& backward_pass(Matrix &input) {
            this->d = &input;
            
            // Matrix::matMul(weightsT, input, this->delta);
            Matrix::matMulT0(weights, input, this->delta);

            // std::cout << "Dense::backward_pass: output" << std::endl;
            // std::cout << this->delta.toString() << std::endl;

            return this->delta;
        }

        virtual void update(float learning_rate);// {
            // std::cout << "Dense::update: weights" << std::endl;
            // std::cout << this->weights.toString() << std::endl;

            // Matrix::matMul(*(this->d), this->inputT, this->dweights);
            // Matrix::matMulT1(*(this->d), *(this->inputT), this->dweights);
            // Matrix::matScalarMul(learning_rate, this->dweights, this->dweights);
            // Matrix::matSub(this->weights, this->dweights, this->weights);
            // Matrix::matT(this->weights, this->weightsT);

            // std::cout << "Dense::update: dweights" << std::endl;
            // std::cout << this->dweights.toString() << std::endl;
            // std::cout << "Dense::update: weights" << std::endl;
            // std::cout << this->weights.toString() << std::endl;
        // };

};

#endif