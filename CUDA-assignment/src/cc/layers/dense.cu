#include "dense.hh"

#define BLOCK_SIZE 32
#define BLOCK_ROUND_DOWN(x) (x & (~(BLOCK_SIZE-1)))
#define BLOCK_ROUND_UP(x) ((x + BLOCK_SIZE-1) & (~(BLOCK_SIZE-1))) 


__global__ void matMulUpdateCUDA(float *A, float *B, float *C, int M, int N, int K, float lr) {

    float Cvalue = 0.0f;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.x * blockDim.x + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];
    
    for(int i = 0; i < K; i+= BLOCK_SIZE) {

        A_shared[threadIdx.y][threadIdx.x] = A[K * y + i + threadIdx.x];        
        B_shared[threadIdx.x][threadIdx.y] = B[K * z + i + threadIdx.x];

        __syncthreads();

        #pragma unroll
        for(int j = 0; j < BLOCK_SIZE; ++j) {
            Cvalue += A_shared[threadIdx.y][j] * B_shared[j][threadIdx.x];
        }

        __syncthreads();
    }
  
    C[y * N + x] = lr* Cvalue;
}

// __global__ 
// void update_weights(float *A, float *B, float *C, int M, int N, int K, float lr) {
    
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
    
//     C[y * N + x] -= lr * matMulT1CUDA(A, B, C, M, N, K);
// }


void Dense::update(float lr) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(BLOCK_ROUND_UP(this->dweights.cols) / dimBlock.x, BLOCK_ROUND_UP(this->dweights.rows) / dimBlock.y);

    matMulUpdateCUDA<<<dimGrid, dimBlock>>>(this->d->buff, this->inputT->buff, this->dweights.buff, this->d->rows, this->inputT->rows, this->inputT->cols, lr);
    Matrix::matSub(this->weights, this->dweights, this->weights);
};