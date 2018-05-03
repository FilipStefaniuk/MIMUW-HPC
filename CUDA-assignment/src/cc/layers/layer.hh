#ifndef __LAYER_H__
#define __LAYER_H__

#include <iostream>
#include <ios>
#include <iomanip>
#include <sstream>
#include <string>
#include "../matrix/matrix.hh"

#define RANDOM_INIT 0x00
#define ONES_INIT   0x01

class Layer {

    protected:

        Matrix output, delta;
        std::string name;

    public:
        
        Layer(int input_size, int output_size, int batch_size, std::string name) 
            : output(output_size, batch_size), 
              delta(input_size, batch_size), 
              name(name) {}

        Layer(Layer &prev, std::string name)
            : Layer(prev.getOutput().getRows(), 
              prev.getOutput().getRows(), 
              prev.getOutput().getCols(), name) {}
        
        virtual ~Layer() {};

        Matrix const & getOutput() const { return this->output; }
        Matrix const & getDelta() const { return this->delta; } 

        std::string info() {
            std::stringstream ss;
            ss << std::left << std::setw(12) << this->name;
            ss << "[" << std::setw(4) << this->delta.getCols() << ", " << std::setw(4) << this->delta.getRows() << "]" << std::string(4, ' ');
            ss << "[" << std::setw(4) << this->output.getCols() << ", " << std::setw(4) << this->output.getRows() <<  "]";
            return ss.str();
        }

        virtual void initialize(int flag) {};
        
        virtual Matrix & forward_pass(Matrix &input)=0;

        virtual Matrix & backward_pass(Matrix &output)=0;

        virtual void update(float lr) {};


};

#endif