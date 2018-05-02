#include "tanh.hh"

//-----------------------------------------------------------------------------
//                            TANH ACTIVATION                                       
//-----------------------------------------------------------------------------

__global__
void activationTanhCUDA(float *A, float *B, float *C, int M, int N) {

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    float tmp = tanh(A[y * N + x]);
    B[y * N + x] = tmp;
    C[y * N + x] = 1 - tmp;
}

void Tanh::activation(Matrix &input, Matrix &output, Matrix &output_prime) {

    int cols = BLOCK_ROUND_UP(input.getCols());
    int rows = BLOCK_ROUND_UP(input.getRows());

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(cols / dimBlock.x, rows / dimBlock.y);

    activationTanhCUDA<<<dimGrid, dimBlock>>>
    (input.buff, output.buff, output_prime.buff, rows, cols);
}
