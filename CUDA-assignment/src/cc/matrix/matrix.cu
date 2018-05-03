#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <random>
#include "matrix.hh"

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) {
    cudaMalloc((void**)&(this->buff), BLOCK_ROUND_UP(rows) * BLOCK_ROUND_UP(cols) * sizeof(float));
}

Matrix::~Matrix() {
    cudaFree(this->buff);
}

std::ostream& operator<<(std::ostream& stream, Matrix const &matrix) {
    return stream << "Matrix[" << matrix.rows << ", " << matrix.cols << "]";
}

//-----------------------------------------------------------------------------
//                            MATRIX MULTIPLICATION                                       
//-----------------------------------------------------------------------------

__global__
void matMulCUDA(float *A, float *B, float *C, int M, int N, int K) {

    float Cvalue = 0.0f;

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
     
    C[y * N + x] = Cvalue;
}

__global__
void matMulLeftTCUDA(float *A, float *B, float *C, int M, int N, int K) {

    float Cvalue = 0.0f;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.y * blockDim.y + threadIdx.x;

    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];
    
    for(int i = 0; i < K; i+= BLOCK_SIZE) {

        A_shared[threadIdx.y][threadIdx.x] = A[(i + threadIdx.y) * M + z];        
        B_shared[threadIdx.y][threadIdx.x] = B[(i + threadIdx.y) * N + x];

        __syncthreads();

        #pragma unroll
        for(int j = 0; j < BLOCK_SIZE; ++j) {
            Cvalue += A_shared[j][threadIdx.y] * B_shared[j][threadIdx.x];
        }

        __syncthreads();
    }

    C[y * N + x] = Cvalue;
}

__global__ void matMulRightTCUDA(float *A, float *B, float *C, int M, int N, int K) {

    float Cvalue = 0.0f;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int z = blockIdx.x * blockDim.x + threadIdx.y;

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

void Matrix::matMul(Matrix const &A, Matrix const &B, Matrix &C, int mode) {

    int b_rows = BLOCK_ROUND_UP(C.getRows());
    int b_cols = BLOCK_ROUND_UP(C.getCols());

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(b_cols / dimBlock.x, b_rows / dimBlock.y);

    if (mode == LEFT_T) {
    
        matMulLeftTCUDA<<<dimGrid, dimBlock>>>
        (A.buff, B.buff, C.buff, b_rows, b_cols, BLOCK_ROUND_UP(B.rows));
    
    } else if (mode == RIGHT_T) {
    
        matMulRightTCUDA<<<dimGrid, dimBlock>>>
        (A.buff, B.buff, C.buff, b_rows, b_cols, BLOCK_ROUND_UP(B.cols));
    
    } else {
    
        matMulCUDA<<<dimGrid, dimBlock>>>
        (A.buff, B.buff, C.buff, b_rows, b_cols, BLOCK_ROUND_UP(B.rows));
    
    }
}

//-----------------------------------------------------------------------------
//                            MATRIX SUBTRACTION                                       
//-----------------------------------------------------------------------------

__global__ 
void matSubCUDA(float *A, float *B, float *C, int M, int N) {
    
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    C[y * N + x] = A[y * N + x] - B[y * N + x];
}

void Matrix::matSub(Matrix const &A, Matrix const &B, Matrix &C) {

    int b_rows = BLOCK_ROUND_UP(C.getRows());
    int b_cols = BLOCK_ROUND_UP(C.getCols());

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(b_cols / dimBlock.x, b_rows / dimBlock.y);

    matSubCUDA<<<dimGrid, dimBlock>>>
    (A.buff, B.buff, C.buff, b_rows, b_cols);
}

//-----------------------------------------------------------------------------
//                            MATRIX EL MUL                                       
//-----------------------------------------------------------------------------

__global__ 
void matElMulCUDA(float *A, float *B, float *C, int M, int N) {
    
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    C[y * N + x] = A[y * N + x] * B[y * N + x];
}

__global__ 
void matScalarMulCUDA(float const e, float *A, float *B, int M, int N) {
    
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    B[y * N + x] = e * A[y * N + x];
}

void Matrix::matElMul(Matrix const &A, Matrix const &B, Matrix &C) {

    int b_rows = BLOCK_ROUND_UP(C.getRows());
    int b_cols = BLOCK_ROUND_UP(C.getCols());

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(b_cols / dimBlock.x, b_rows / dimBlock.y);

    matElMulCUDA<<<dimGrid, dimBlock>>>(A.buff, B.buff, C.buff, b_rows, b_cols);
}

void Matrix::matElMul(float const x, Matrix const &A, Matrix &B) {

    int b_rows = BLOCK_ROUND_UP(A.getRows());
    int b_cols = BLOCK_ROUND_UP(A.getCols());

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(b_cols / dimBlock.x, b_rows / dimBlock.y);

    matScalarMulCUDA<<<dimGrid, dimBlock>>>(x, A.buff, B.buff, b_rows, b_cols);
}

//-----------------------------------------------------------------------------
//                            INIT                                       
//-----------------------------------------------------------------------------

void Matrix::init() {

    int b_rows = BLOCK_ROUND_UP(this->rows);
    int b_cols = BLOCK_ROUND_UP(this->cols);

    float *tmp_buff = (float*) calloc(b_rows * b_cols, sizeof(float));
    
    std::mt19937 rng;
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    for(int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            tmp_buff[i * b_cols + j] = distribution(rng);
        }
    }

    cudaMemcpy(this->buff, tmp_buff, b_rows * b_cols * sizeof(float), cudaMemcpyHostToDevice);
    free(tmp_buff);
}

void Matrix::init(float val) {

    int b_rows = BLOCK_ROUND_UP(this->rows);
    int b_cols = BLOCK_ROUND_UP(this->cols);

    float *tmp_buff = (float*) calloc(b_rows * b_cols, sizeof(float));

    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            tmp_buff[i * b_cols + j] = val;
        }
    }

    cudaMemcpy(this->buff, tmp_buff, b_rows * b_cols * sizeof(float), cudaMemcpyHostToDevice);
    free(tmp_buff);
}

void Matrix::init(float *buff) {

    int b_rows = BLOCK_ROUND_UP(this->rows);
    int b_cols = BLOCK_ROUND_UP(this->cols);

    if (this->cols == b_cols && this->rows == b_rows) {
        cudaMemcpy(this->buff, buff, b_rows * b_cols * sizeof(float), cudaMemcpyHostToDevice);
        return;
    }

    float *tmp_buff = (float*) calloc(b_rows * b_cols, sizeof(float));

    if (this->cols == b_cols) {
    
        memcpy(tmp_buff, buff, this->rows * this->cols * sizeof(float));        
    
    } else {

        for (int i = 0; i < this->rows; ++i) {
            memcpy(tmp_buff + b_cols * i, buff + this->cols * i, this->cols * sizeof(float));
        }
    }

    cudaMemcpy(this->buff, tmp_buff, b_rows * b_cols * sizeof(float), cudaMemcpyHostToDevice);
    free(tmp_buff);
}

//-----------------------------------------------------------------------------
//                                                                   
//-----------------------------------------------------------------------------

// Used only for testing
bool Matrix::operator==(Matrix const &other) const {

    if (this->rows != other.rows || this->cols != other.cols) {
        return false;
    }

    unsigned size = BLOCK_ROUND_UP(this->rows) * BLOCK_ROUND_UP(this->cols);

    float *a = new float[size];
    float *b = new float[size];

    cudaMemcpy(a, this->buff, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, other.buff, size * sizeof(float), cudaMemcpyDeviceToHost);

    bool equal = true;
    for (int i = 0; i < size; ++i) {
        if (a[i] != b[i]) {
            equal = false;
            break;
        }
    }

    delete a;
    delete b;

    return equal;
}

// Used only for testing
std::string Matrix::toString() const {

    std::stringstream ss;
    ss << *this << std::endl;
    ss << std::fixed << std::setprecision(2);
    
    float *a = new float[BLOCK_ROUND_UP(this->rows) * BLOCK_ROUND_UP(this->cols)];

    cudaMemcpy(a, this->buff, BLOCK_ROUND_UP(this->rows) * BLOCK_ROUND_UP(this->cols) * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < BLOCK_ROUND_UP(this->rows); ++i) {
        for (int j = 0; j < BLOCK_ROUND_UP(this->cols); ++j) {
            
            if (j) {
                ss << " ";
            }
            
            ss << a[BLOCK_ROUND_UP(this->cols) * i + j];
        }
        ss << std::endl;
    }

    delete a;

    return ss.str();
}