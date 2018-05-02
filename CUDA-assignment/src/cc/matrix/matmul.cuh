#ifndef __MATMUL_CUH__
#define __MATMUL_CUH__

// #pragma once

#define BLOCK_SIZE 32
#define BLOCK_ROUND_DOWN(x) (x & (~(BLOCK_SIZE-1)))
#define BLOCK_ROUND_UP(x) ((x + BLOCK_SIZE-1) & (~(BLOCK_SIZE-1))) 


__global__
void matMulCUDA(float *A, float *B, float *C, int M, int N, int K) {

    float Cvalue = 0.0f;
    // int a = K % BLOCK_SIZE;
    // int b = BLOCK_ROUND_DOWN(K);
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];
    
    for(int i = 0; i < K; i+= BLOCK_SIZE) {

        A_shared[threadIdx.y][threadIdx.x] = A[K * y + i + threadIdx.x];        
        B_shared[threadIdx.y][threadIdx.x] = B[(i + threadIdx.y) * N + x];

        __syncthreads();

        #pragma unroll
        for(int j = 0; j < BLOCK_SIZE; ++j) {
            Cvalue += A_shared[threadIdx.y][j] * B_shared[j][threadIdx.x];
        }

        __syncthreads();
    }

    // if (a) {

    //     A_shared[threadIdx.y][threadIdx.x] = threadIdx.x < a ? A[K * y + b + threadIdx.x] : 0;        
    //     B_shared[threadIdx.y][threadIdx.x] = threadIdx.y < a ? B[(b + threadIdx.y) * N + x] : 0;

    //     __syncthreads();

    //     #pragma unroll
    //     for(int j = 0; j < BLOCK_SIZE; ++j) {
    //         Cvalue += A_shared[threadIdx.y][j] * B_shared[j][threadIdx.x];
    //     }

    //     __syncthreads();
    // }

    // if (y < M && x < N) {    
        C[y * N + x] = Cvalue;
    // }
}


__global__
void matMulT0CUDA(float *A, float *B, float *C, int M, int N, int K) {

    float Cvalue = 0.0f;
    // int a = K % BLOCK_SIZE;
    // int b = BLOCK_ROUND_DOWN(K);
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.x;

    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];
    
    for(int i = 0; i < K; i+= BLOCK_SIZE) {

        A_shared[threadIdx.y][threadIdx.x] = A[(i + threadIdx.y) * M + z]; //A[K * y + i + threadIdx.x];        
        B_shared[threadIdx.y][threadIdx.x] = B[(i + threadIdx.y) * N + x];

        __syncthreads();

        #pragma unroll
        for(int j = 0; j < BLOCK_SIZE; ++j) {
            Cvalue += A_shared[j][threadIdx.y] * B_shared[j][threadIdx.x];
            // Cvalue += A_shared[j][threadIdx.y];
        }

        __syncthreads();
    }

    // if (a) {

    //     if (y < M && x < N) {
    //         A_shared[threadIdx.y][threadIdx.x] = threadIdx.y < a ? A[(b + threadIdx.y) * M + blockIdx.y * blockDim.y + x] : 0; //A[K * y + b + threadIdx.x] : 0;        
    //         B_shared[threadIdx.y][threadIdx.x] = threadIdx.y < a ? B[(b + threadIdx.y) * N + x] : 0;
    //     } else {
    //         A_shared[threadIdx.y][threadIdx.x] = 0;
    //         B_shared[threadIdx.y][threadIdx.x] = 0;
    //     }
    //     __syncthreads();

    //     #pragma unroll
    //     for(int j = 0; j < a; ++j) {
    //         Cvalue += A_shared[j][threadIdx.y] * B_shared[j][threadIdx.x];
    //     }

    //     __syncthreads();
    // }

    // if (y < M && x < N) {    
        C[y * N + x] = Cvalue;
    // }
}

// __device__ __forceinline__
__global__ void matMulT1CUDA(float *A, float *B, float *C, int M, int N, int K) {

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
  
    C[y * N + x] = Cvalue;
}

#endif