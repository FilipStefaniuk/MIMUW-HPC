#ifndef __DENSE_HH__
#define __DENSE_HH__

#include "layer.hh"
#include <iostream>

class Dense : public Layer {
    private:

        Matrix weights, dweights, bias, dbias;
        Matrix *d, *inputT;

    public:

        Dense(int input_size, int batch_size, int size) 
            : Layer(input_size, size, batch_size, "DENSE"),
              weights(size, input_size),
              dweights(size, input_size),
              bias(size, 1), dbias(size, 1) {}

        Dense(Layer &prev, int size)
            : Dense(prev.getOutput().getRows(), 
              prev.getOutput().getCols(), size) {}

        ~Dense() {}

        void initialize(int flag) {
            if (flag) {
                this->weights.init();
                this->bias.init();
                this->dbias.init();
            } else {
                this->weights.init(1.);
                this->bias.init(1.);
                this->dbias.init(1.);
            }
        }

        Matrix& forward_pass(Matrix &input) {

            // std::cout << "INPUT DENSE" << std::endl;
            // std::cout << input.toString() << std::endl;
            // std::cout << "---------------------" << std::endl;

            std::cout << "WEIGHTS" << std::endl;
            std::cout << this->weights.toString() << std::endl;
            std::cout << "---------------------" << std::endl;


            Matrix::matMul(this->weights, input, this->output, 0);
            Matrix::vecAdd(this->output, this->bias, this->output);
            this->inputT = &input;
            
            return this->output;
        }

        Matrix& backward_pass(Matrix &input) {
            this->d = &input;
            Matrix::matMul(weights, input, this->delta, LEFT_T);

            return this->delta;
        }

        void update(float lr) {
            // std::cout << "OLD WEIGHT" << std::endl;
            // std::cout << this->weights.toString() << std::endl;

            // std::cout << "DELTA" << std::endl;
            // std::cout << this->d->toString() << std::endl;

            // std::cout << "BIAS" << std::endl;
            // std::cout << this->bias.toString() << std::endl;

            // std::cout << "INPUT T" << std::endl;
            // std::cout << this->inputT->toString() << std::endl;

            Matrix::matMul(*(this->d), *(this->inputT), this->dweights, RIGHT_T);
            Matrix::matElMul(lr / this->d->getCols(), this->dweights, this->dweights);
            Matrix::matSub(this->weights, this->dweights, this->weights);
            
            // std::cout << "DBIAS" << std::endl;
            // std::cout << this->dbias.toString() << std::endl;

            Matrix::rowSum(*(this->d), this->dbias);
            Matrix::matElMul(lr / this->d->getCols(), this->dbias, this->dbias);
            Matrix::matSub(this->bias, this->dbias, this->bias);
            
            // std::cout << "WEIGHTS" << std::endl;
            // std::cout << this->weights.toString() << std::endl;
            // std::cout << "---------------------" << std::endl;
        }

};

#endif