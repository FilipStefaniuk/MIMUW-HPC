#include "relu.hh"

//-----------------------------------------------------------------------------
//                            RELU ACTIVATION                                       
//-----------------------------------------------------------------------------

__global__
void activationReLUCUDA(float *A, float *B, float *C, int M, int N) {

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    float tmp = A[y * N + x];
    B[y * N + x] = tmp > 0 ? tmp : 0;
    C[y * N + x] = tmp > 0 ? 1 : 0;
}

void ReLU::activation(Matrix &input, Matrix &output, Matrix &output_prime) {

    int cols = BLOCK_ROUND_UP(input.getCols());
    int rows = BLOCK_ROUND_UP(input.getRows());

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(cols / dimBlock.x, rows / dimBlock.y);

    activationReLUCUDA<<<dimGrid, dimBlock>>>
    (input.buff, output.buff, output_prime.buff, rows, cols);
}
