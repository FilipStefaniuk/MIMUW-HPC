#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define A_WIDTH 2000
#define A_HEIGHT 1000
#define B_WIDTH 1500

#define BLOCK_SIZE 32

#define MATRIX_ZEROS 0x0
#define MATRIX_RANDOM 0x1

//-----------------------------------------------------------------------------
//                           Matrix Structure
//-----------------------------------------------------------------------------

typedef struct {
    int width;
    int height;
    int stride;
    float * elements;
} Matrix;

Matrix* genMatrix(int width, int height, int flags) {
    Matrix *A = (Matrix*) malloc(sizeof(Matrix));
    A->width = width;
    A->height = height;
    A->elements = (float*) malloc(width * height * sizeof(float));

    if (MATRIX_RANDOM & flags) {
        for (int i = 0; i < width * height; ++i) {
            A->elements[i] = rand() / (float) RAND_MAX;
        }
    } else {
        memset(A->elements, 0.0f, width * height);
    }

    return A;
}

float verify(Matrix *A, Matrix *B) {
    float error = 0.0f;
    for (int i = 0; i < A->width * A->height; ++i) {
        float tmp = fabs(A->elements[i] - B->elements[i]);
        if (tmp > error) {
            error = tmp;
        }
    }
    return error;
}

void freeMatrix(Matrix *A) {
    free(A->elements);
    free(A);
}

//-----------------------------------------------------------------------------
//                           Multiplication CPU 
//-----------------------------------------------------------------------------

void matMul(const Matrix *A, const Matrix *B, Matrix *C) {
    for (int i = 0; i < A->height; ++i) {
        for (int j = 0; j < B->width; ++j) {
            C->elements[B->width * i + j] = 0;
            for (int k = 0; k < A->width; ++k) {
                C->elements[B->width * i + j] += 
                    A->elements[A->width * i + k] * B->elements[B->width * k + j]; 
            }
        }
    }
}

void computeCPU(Matrix *A, Matrix *B, Matrix *C) {
    
    struct timespec cpu_start, cpu_stop;

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start);
    matMul(A, B, C);
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_stop);
    
    double result = (cpu_stop.tv_sec - cpu_start.tv_sec) * 1e3
                     + (cpu_stop.tv_nsec - cpu_start.tv_nsec) / 1e6;    
    
    printf( "Execution time on CPU:  %3.1f ms\n", result);
}

//-----------------------------------------------------------------------------
//               Multiplication CUDA (using global memory)
//-----------------------------------------------------------------------------

__global__ void matMulGPU1(const Matrix A, const Matrix B, Matrix C) {

    float Cvalue = 0;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A.height && col < B.width) {

        for (int e = 0; e < A.width; ++e)
            Cvalue += A.elements[row * A.width + e]
                    * B.elements[e * B.width + col];

        C.elements[row * C.width + col] = Cvalue;
    }
}

void computeGPUglobalMem(Matrix *A, Matrix *B, Matrix *C) {

    Matrix dev_A;
    dev_A.width = A->width; 
    dev_A.height = A->height;
    size_t size = A->width * A->height * sizeof(float);
    cudaMalloc(&dev_A.elements, size);
    cudaMemcpy(dev_A.elements, A->elements, size, cudaMemcpyHostToDevice);
    
    Matrix dev_B;
    dev_B.width = B->width; 
    dev_B.height = B->height;
    size = B->width * B->height * sizeof(float);
    cudaMalloc(&dev_B.elements, size);
    cudaMemcpy(dev_B.elements, B->elements, size, cudaMemcpyHostToDevice);

    Matrix dev_C;
    dev_C.width = C->width; 
    dev_C.height = C->height;
    size = C->width * C->height * sizeof(float);
    cudaMalloc(&dev_C.elements, size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B->width / dimBlock.x, A->height / dimBlock.y);
    
    cudaEventRecord(start, 0);
    matMulGPU1<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop);
    printf("Execution time on GPU (global memory):  %3.1f ms\n", elapsedTime);

    cudaMemcpy(C->elements, dev_C.elements, size, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_A.elements);
    cudaFree(dev_B.elements);
    cudaFree(dev_C.elements);
}

//-----------------------------------------------------------------------------
//               Multiplication CUDA (using shared memory)
//-----------------------------------------------------------------------------

void computeGPUsharedMem(Matrix *A, Matrix *B, Matrix *C) {
    
    // Load A and B to device memory
    Matrix dev_A;
    dev_A.width = dev_A.stride = A->width; 
    dev_A.height = A->height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    
    Matrix d_B;
    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
    cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C->elements, dev_C.elements, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dev_A.elements);
    cudaFree(dev_B.elements);
    cudaFree(dev_C.elements);
}

//-----------------------------------------------------------------------------
//               
//-----------------------------------------------------------------------------

int main() {

    Matrix *A = genMatrix(A_WIDTH, A_HEIGHT, MATRIX_RANDOM);
    Matrix *B = genMatrix(B_WIDTH, A_WIDTH, MATRIX_RANDOM);
    
    Matrix *C1 = genMatrix(B_WIDTH, A_HEIGHT, MATRIX_ZEROS);
    Matrix *C2 = genMatrix(B_WIDTH, A_HEIGHT, MATRIX_ZEROS);
    Matrix *C3 = genMatrix(B_WIDTH, A_HEIGHT, MATRIX_ZEROS);

    computeCPU(A, B, C1);
    computeGPUglobalMem(A, B, C2);
    computeGPUsharedMem(A, B, C3);

    // for (int i = 0; i < C1->width * C1->height; ++i) {
    //     printf("%f | %f\n", C1->elements[i], C2->elements[i]);
    // }

    float max_diff = verify(C1, C2);
    printf("Maximum difference (CPU, GPU_glob) %f\n", max_diff);

    free(A);
    free(B);
    free(C1);
    free(C2);
    free(C3);
    return 0;
}

