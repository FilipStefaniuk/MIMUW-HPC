#include <iostream>
#include <cmath>
#include <chrono>
#include "model.hh"

//-----------------------------------------------------------------------------
//                            ACCURACY                                       
//-----------------------------------------------------------------------------

float Model::accuracy(Matrix &pred_vals, Matrix &true_vals) {
    
    int sum = 0;

    for(int i = 0; i < pred_vals.getCols(); ++i) {
        
        float pred_max = pred_vals.buff[i];
        float true_max = true_vals.buff[i];

        int pred_arg_max = 0, true_arg_max = 0;

        for(int j = 0; j < pred_vals.getRows(); ++j) {
            float pred_val = pred_vals.buff[j * pred_vals.getCols() + i];
            float true_val = true_vals.buff[j * true_vals.getCols() + i];
            
            if (pred_val > pred_max) {
                pred_max = pred_val;
                pred_arg_max = j;
            }

            if (true_val > true_max) {
                true_max = true_val;
                true_arg_max = j;
            }
        }

        sum += pred_arg_max == true_arg_max;
    }

    return ((float) sum) / (float) pred_vals.getCols();
}

//-----------------------------------------------------------------------------
//                            COST                                       
//-----------------------------------------------------------------------------

float Model::crossEntropyCost(Matrix &pred_vals, Matrix &true_vals) {
    
    float sum = 0;

    for (int i = 0; i < pred_vals.getCols(); ++i) {
        for (int j = 0; j  < pred_vals.getRows(); ++j) {
            if (true_vals.buff[true_vals.getCols() * j + i] != 0.0f) {
            sum -= logf(pred_vals.buff[pred_vals.getCols() * j + i]);
            }
        }
    }

    return sum / pred_vals.getCols();
}

//-----------------------------------------------------------------------------
//                            FIT                                       
//-----------------------------------------------------------------------------

void Model::fit(float *data_x, float *data_y, int len,  int epochs, 
                float learning_rate, float eps, int random) {

    // Initialize input & output
    this->input.init(data_x);
    this->output.init(data_y);

    // Initialize layers
    for (Layer *l : this->layers) {
        l->initialize(1);
    }

    for (int i = 0; i < epochs; ++i) {
        
        auto start = std::chrono::steady_clock::now();

        // Forward pass
        Matrix *input = &this->input;
        for (Layer *layer : this->layers) {
            input = &layer->forward_pass(*input);
            
            // std::cout << "OUTPUT VALUES" << std::endl;
            // std::cout << input->toString() << std::endl;
            // std::cout << "---------------------" << std::endl;
        }

        // Loss Function
        float tmp_cost = Model::crossEntropyCost(*input, this->output);
        float acc = Model::accuracy(*input, this->output);

        // Delta
        Matrix::matSub(*input, this->output, this->delta);

        // Backward pass
        Matrix *delta = &this->delta;
        for (auto it = this->layers.rbegin(); it != this->layers.rend(); ++it) {
            delta = &(*it)->backward_pass(*delta);

            // std::cout << "DELTA" << std::endl;
            // std::cout << delta->toString() << std::endl;
            // std::cout << "---------------------" << std::endl;
        }

        // Update
        for (Layer * layer : this->layers) {
            layer->update(learning_rate);
        }

        auto end = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << std::fixed << "epoch " << i + 1 << "/" << epochs << "\t" 
                  << std::setprecision(3) << "time: " << elapsedTime.count() << " ms, "
                  << "cost: " << tmp_cost << ", "
                  << std::setprecision(2) << "accuracy: " << acc << std::endl;
    }
}