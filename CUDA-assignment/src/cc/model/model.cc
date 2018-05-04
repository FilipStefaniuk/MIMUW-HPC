#include <iostream>
#include <cmath>
#include <chrono>
#include <cstring>
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
//                            GET BATCH                                       
//-----------------------------------------------------------------------------

void getBatch(Matrix &input, Matrix &batch, int n) {

    int cols = batch.getCols();

    for (int i = 0; i < input.getRows(); ++i) {
        memcpy(batch.buff + cols * i, input.buff + input.getCols() * i + n * cols, cols * sizeof(float));
    }
}

//-----------------------------------------------------------------------------
//                            FIT                                       
//-----------------------------------------------------------------------------

float Model::fit(float *data_x, float *data_y, int len,  int epochs, 
                float learning_rate, float eps, int random) {

    // Initialize input & output
    Matrix data_X(this->input_size, len);
    Matrix data_Y(this->output_size, len);
    data_X.init(data_x);
    data_Y.init(data_y);

    // Initialize layers
    for (Layer *l : this->layers) {
        l->initialize(random);
    }
    
    int i;
    float sum_acc = 0;

    for (i = 0; i < epochs; ++i) {
        
        int j;
        float acc = 0, cost = 0;
        auto start = std::chrono::steady_clock::now();

        for (j = 0; j < len / this->batch_size; ++j) {

            getBatch(data_X, this->input, j);
            getBatch(data_Y, this->output, j);

            // std::cout << "DATA_Y" << std::endl;
            // std::cout << data_Y.toString() << std::endl;
            // std::cout << "---------------------" << std::endl;

            // std::cout << "BATCH_X" << std::endl;
            // std::cout << this->input.toString() << std::endl;
            // std::cout << "---------------------" << std::endl;

            // std::cout << "BATCH_Y" << std::endl;
            // std::cout << this->output.toString() << std::endl;
            // std::cout << "---------------------" << std::endl;

            // Forward pass
            Matrix *input = &this->input;
            for (Layer *layer : this->layers) {
                input = &layer->forward_pass(*input);
                
            }
            
            // std::cout << "OUTPUT VALUES" << std::endl;
            // std::cout << input->toString() << std::endl;
            // std::cout << "---------------------" << std::endl;

            // std::cout << "CORRECT VALUES" << std::endl;
            // std::cout << this->output.toString() << std::endl;
            // std::cout << "---------------------" << std::endl;

            // Loss Function
            cost += Model::crossEntropyCost(*input, this->output);
            acc += Model::accuracy(*input, this->output);
            

            // Delta
            Matrix::matSub(*input, this->output, this->delta);

            // Backward pass
            Matrix *delta = &this->delta;
            for (auto it = this->layers.rbegin(); it != this->layers.rend(); ++it) {
                // std::cout << "DELTA" << std::endl;
                // std::cout << delta->toString() << std::endl;
                // std::cout << "---------------------" << std::endl;
                delta = &(*it)->backward_pass(*delta);
            }

            // Update
            for (Layer * layer : this->layers) {
                layer->update(learning_rate);
            }
        }
        
        auto end = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        cost /= j;
        acc /= j;
        
        sum_acc += acc;

        std::cout << std::fixed << "epoch " << i + 1 << "/" << epochs << "\t" 
                  << std::setprecision(3) << "time: " << elapsedTime.count() << " ms, "
                  << "cost: " << cost << ", "
                  << std::setprecision(2) << "accuracy: " << acc << std::endl;
        
        if (cost < eps) {
            ++i;
            break;
        }
    }

    return sum_acc / i;
}