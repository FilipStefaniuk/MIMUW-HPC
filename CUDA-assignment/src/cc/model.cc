#include <iostream>
#include <string>
#include "model.hh"
#include <thread>
#include "layers/input.hh"
#include "initializers/buffer_initializer.hh"
#include "initializers/random_initializer.hh"
#include "initializers/const_initializer.hh"

#ifndef NDEBUG
    const bool debug = true;
#else
    const bool debug = false;
#endif



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

void Model::fit(float *data_x, float *data_y, int len,  int epochs, float learning_rate, float eps, int random) {
    

    // Initialization
    //-------------------------------------------------------------------------
    Initializer *initializer;
    
    initializer = random ? (Initializer*) new RandomInitializer() : (Initializer*) new ConstInitializer(0.5f);

    this->input.initialize(data_x);
    this->output.initialize(data_y);

    for (auto it = std::next(this->layers.begin()); it != this->layers.end(); ++it) {
        (*it)->initialize(*initializer);
    }

    delete initializer;

    for (int i = 0; i < epochs; ++i) {

        std::cout << "EPOCH: " << i << std::endl;
        // float cost = 0.0f;
        
        // Change that (last batch may be cut)
        // for (int b = 0; b < len/batch_size; ++b) {

            // this->input.initialize(&data_x[b*this->batch_size*this->input_size]);
            // this->output.initialize(&data_y[b*this->batch_size*this->output_size]);

            // std::cout << "Model::fit: input batch" << std::endl;
            // std::cout << this->input.toString() << std::endl;

            // std::cout << "Model::fit: out batch" << std::endl;
            // std::cout << this->output.toString() << std::endl;
        
            // Forward pass
            //-------------------------------------------------------------------------
            Matrix *input = &this->input;
            for (Layer *layer : this->layers) {
                input = &layer->forward_pass(*input);
            }

            // Loss Function
            //-------------------------------------------------------------------------
            float tmp_cost = Matrix::cost(*input, this->output);
            // cost += tmp_cost;

            Matrix::matSub(*input, this->output, this->delta);

            
            std::cout << "Model::fit: tmp cost: " << tmp_cost << std::endl;
            // std::cout << "Model::fit: pred_vals" << std::endl;
            // std::cout << input->toString() << std::endl;
            // std::cout << "Model::fit: true_vals" << std::endl;
            // std::cout << this->output.toString() << std::endl;
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

            // std::thread updates[this->layers.size()];
            // for (unsigned j = 0; j < this->layers.size(); ++j) {
            //     updates[j] = std::thread(&Layer::update, this->layers[j], learning_rate);
            // }

            // for (unsigned j = 0; j < this->layers.size(); ++j) {
            //     updates[j].join();
            // }
            
            for (Layer * layer : this->layers) {
                layer->update(learning_rate);
            }
        // }

        // cost /= (len/batch_size);
        // std::cout << "Model::fit: cost: " << cost << std::endl;
    }
}

