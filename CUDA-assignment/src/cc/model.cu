#include <iostream>
#include <string>
#include "model.hh"
#include <thread>
#include "layers/input.hh"
#include "initializers/buffer_initializer.hh"
#include "initializers/random_initializer.hh"
#include "initializers/const_initializer.hh"

#define BLOCK_SIZE 32
#define BLOCK_ROUND_UP(x) ((x + BLOCK_SIZE-1) & (~(BLOCK_SIZE-1))) 

//-----------------------------------------------------------------------------
//                            ACCURACY                                       
//-----------------------------------------------------------------------------

__global__
void accuracyCUDA(float *A, float *B, int *C, int M, int N) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;

    float pred_max = A[x], true_max = B[x];
    int pred_arg_max = 0, true_arg_max = 0;

    for(int j = 0; j < M; ++j) {
        float pred_val = A[j * N + x];
        float true_val = B[j * N + x];
        
        if (pred_val > pred_max) {
            pred_max = pred_val;
            pred_arg_max = j;
        }

        if (true_val > true_max) {
            true_max = true_val;
            true_arg_max = j;
        }
    }

    C[x] = pred_arg_max == true_arg_max;
}

float accuracy(Matrix &pred_vals, Matrix &true_vals) {

    int correct_sum = 0;
    
    int *dev_correct;
    int *correct = new int[pred_vals.cols];

    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(BLOCK_ROUND_UP(pred_vals.cols) / BLOCK_SIZE);
    
    cudaMalloc((void**)&dev_correct, BLOCK_ROUND_UP(pred_vals.cols) * sizeof(int));
    
    accuracyCUDA<<<dimGrid, dimBlock>>>
    (pred_vals.buff, true_vals.buff, dev_correct, pred_vals.rows, BLOCK_ROUND_UP(pred_vals.cols));
    
    cudaMemcpy(correct, dev_correct, pred_vals.cols * sizeof(int), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < pred_vals.cols; ++i) {
        correct_sum += correct[i];
    }

    delete correct;
    cudaFree(dev_correct);

    return ((float) correct_sum) / (float) pred_vals.cols;
}

//-----------------------------------------------------------------------------
//                            COST                                       
//-----------------------------------------------------------------------------

__global__
void crossEntropyCostCUDA(float *A, float *B, float *C, int M, int N) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    float cost = 0;

    for (int i = 0; i < M; ++i) {
        if (B[i * N + x] != 0) {
            cost -= logf(A[i * N + x]);
        } 
    }

    C[x] = cost;
}

float crossEntropyCost(Matrix &pred_vals, Matrix &true_vals) {
    
    float cost_sum = 0.0f;

    float *dev_cost;
    float *cost = new float[pred_vals.cols];
    
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(BLOCK_ROUND_UP(pred_vals.cols) / BLOCK_SIZE);

    cudaMalloc((void**)&dev_cost, BLOCK_ROUND_UP(pred_vals.cols) * sizeof(int));

    crossEntropyCostCUDA<<<dimGrid, dimBlock>>>
    (pred_vals.buff, true_vals.buff, dev_cost, pred_vals.rows, BLOCK_ROUND_UP(pred_vals.cols));

    cudaMemcpy(cost, dev_cost, pred_vals.cols * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < pred_vals.cols; ++i) {
        cost_sum += cost[i];
    }

    delete cost;
    cudaFree(dev_cost);

    return cost_sum / (float) pred_vals.cols;
}

//-----------------------------------------------------------------------------
//                            FIT                                       
//-----------------------------------------------------------------------------

void Model::fit(float *data_x, float *data_y, int len,  int epochs, 
                float learning_rate, float eps, int random) {

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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < epochs; ++i) {

        // std::cout << "EPOCH: " << i << std::endl;
        // float cost = 0.0f;
        
        cudaEventRecord( start, 0 );

        // Forward pass
        //-------------------------------------------------------------------------
        Matrix *input = &this->input;
        for (Layer *layer : this->layers) {
            input = &layer->forward_pass(*input);
        }

        // Loss Function
        //-------------------------------------------------------------------------
        // float tmp_cost = Matrix::cost(*input, this->output);
        float tmp_cost = crossEntropyCost(*input, this->output);
        float acc = accuracy(*input, this->output);
        // cost += tmp_cost;

        Matrix::matSub(*input, this->output, this->delta);

            
        // std::cout << "Model::fit: tmp cost: " << tmp_cost << std::endl;
        // if (i == 99) {
        // std::cout << "Model::fit: pred_vals" << std::endl;
        // std::cout << input->toString() << std::endl;
        // }
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
        
        for (Layer * layer : this->layers) {
            layer->update(learning_rate);
        }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        std::cout << std::fixed << "epoch " << i + 1 << "/" << epochs << "\t" 
                  << std::setprecision(3) << "time: " << elapsedTime << " ms, "
                  << "cost: " << tmp_cost << ", "
                  << std::setprecision(2) << "accuracy: " << acc << std::endl;
    }
        
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}