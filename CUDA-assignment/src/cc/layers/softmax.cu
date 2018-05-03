#include "softmax.hh"

//-----------------------------------------------------------------------------
//                       SOFTMAX FORWARD PASS                                       
//-----------------------------------------------------------------------------

__global__ 
void softmaxCUDA(float *A, float *B, int M, int N, int NN) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < N) {
        float m = A[x];

        for (int j = 0; j < M; ++j) {
            m = fmax(m, A[NN * j + x]);
        }

        float sum = 0.0f;
        for (int j = 0; j < M; ++j) {
            sum += expf(A[NN * j + x] - m);
        }

        for (int j = 0; j < M; ++j) {
            B[NN * j + x] = expf(A[NN * j + x] - m) / sum;
        }
    }
}

Matrix& Softmax::forward_pass(Matrix &input) {

    // std::cout << "SOFTMAX IN" << std::endl;
    // std::cout << input.toString() << std::endl;
    // std::cout << "---------------------" << std::endl;

    int cols = BLOCK_ROUND_UP(input.getCols());
    int rows = BLOCK_ROUND_UP(input.getRows());

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(cols / dimBlock.x, rows / dimBlock.y);

    softmaxCUDA<<<dimGrid, dimBlock>>>
    (input.buff, this->output.buff, input.getRows(), input.getCols(), cols);

    return this->output;
}
