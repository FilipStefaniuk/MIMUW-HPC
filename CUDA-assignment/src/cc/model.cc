#include <iostream>
#include <string>
#include "model.hh"
#include "layers/input.hh"
#include "initializers/buffer_initializer.hh"
#include "initializers/random_initializer.hh"
#include "initializers/const_initializer.hh"

#ifndef NDEBUG
    const bool debug = true;
#else
    const bool debug = false;
#endif

Model::Model(int input_size, int output_size, int batch_size) 
    : input(input_size, batch_size),
      output(output_size, batch_size),
      delta(output_size, batch_size) {
        layers.push_back(new Input(input_size, batch_size));
}

Model::~Model() {
    for (Layer *layer : this->layers) {
        delete layer;
    }
}

// float Model::evaluate(Matrix &pred_val, Matrix &true_val) {

//     int correct_preds = 0;

//     for (int i = 0; i < pred_val.width; ++i) {
        
//         float max_pred = pred_val.elements[i];
//         float max_true = true_val.elements[i];
//         int arg_max_pred = 0, arg_max_true = 0;
        
//         for(int j = 0; j < pred_val.height; ++j) {
//             if (pred_val.elements[pred_val.width * j + i] > max_pred) {
//                 max_pred = pred_val.elements[pred_val.width * j + i];
//                 arg_max_pred = j;
//             }

//             if (true_val.elements[true_val.width * j + i] > max_true) {
//                 max_true = true_val.elements[true_val.width * j + i];
//                 arg_max_true = j;
//             }        
//         }

//         if (arg_max_pred == arg_max_true) {
//             correct_preds += 1;
//         }
//     }

//     return ((float) correct_preds) / pred_val.width;
// }

void Model::fit(float *data_x, float *data_y, int epochs, float learning_rate, float eps, int random) {
    

    // Initialization
    //-------------------------------------------------------------------------
    Initializer *initializer;
    BufferInitializer data_X(data_x), data_Y(data_y);
    
    initializer = random ? (Initializer*) new RandomInitializer() : (Initializer*) new ConstInitializer(0.5f);

    this->input.initialize(data_X);
    this->output.initialize(data_Y);

    for (auto it = std::next(this->layers.begin()); it != this->layers.end(); ++it) {
        (*it)->initialize(*initializer);
    }

    delete initializer;

    for (int i = 0; i < epochs; ++i) {

        std::cout << "EPOCH: " << i << std::endl;
    
        // Forward pass
        //-------------------------------------------------------------------------
        Matrix *input = &this->input;
        for (Layer *layer : this->layers) {
            input = &layer->forward_pass(*input);
        }

        // Loss Function
        //-------------------------------------------------------------------------
        double cost = Matrix::cost(*input, this->output);
        Matrix::matSub(*input, this->output, this->delta);

        // std::cout << "Model::fit: pred_vals" << std::endl;
        // std::cout << input->toString() << std::endl;
        // std::cout << "Model::fit: true_vals" << std::endl;
        // std::cout << this->output.toString() << std::endl;
        std::cout << "Model::fit: cost: " << cost << std::endl;
        // std::cout << "Model::fit: delta" << std::endl;
        // std::cout << this->delta.toString() << std::endl;
        // std::cout << std::endl;

        // Backward pass
        //-------------------------------------------------------------------------
        Matrix *delta = &this->delta;
        for (auto it = this->layers.rbegin(); it != this->layers.rend(); ++it) {
            delta = &(*it)->backward_pass(*delta);
        }

        // Update Weights
        //-------------------------------------------------------------------------
        for (Layer * layer : this->layers) {
            layer->update(learning_rate);
        }
    }
}

void Model::summary() {

    std::cout << std::string(40, '*') << std::endl;
    std::cout << std::string(10, ' ') << "Model Summary" << std::endl;
    std::cout << std::string(40, '*') <<std::endl;

    for (Layer *layer : this->layers) {
        std::cout << layer->info() << std::endl;
        std::cout << std::string(40, '-') << std::endl;
    }

    std::cout << std::endl;
}