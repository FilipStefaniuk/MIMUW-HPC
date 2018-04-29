#include <sstream>
#include <iomanip>
#include "matrix.hh"

#define BLOCK_SIZE 32
#define BLOCK_ROUND_DOWN(x) (x & (~(BLOCK_SIZE-1)))
#define BLOCK_ROUND_UP(x) ((x + BLOCK_SIZE-1) & (~(BLOCK_SIZE-1))) 


Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) {
    cudaMalloc((void**)&(this->buff), rows * cols * sizeof(float));
}

// Matrix::Matrix(unsigned rows, unsigned cols, float *buff) : rows(rows), cols(cols) {
//     cudaMalloc((void**)&(this->buff), rows * cols * sizeof(float));
//     cudaMemcpy(this->buff, buff, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
// }

Matrix::~Matrix() {
    cudaFree(this->buff);
}

__global__
void matMulCUDA(float *A, float *B, float *C, int M, int N, int K) {

    float Cvalue = 0.0f;
    int a = K % BLOCK_SIZE;
    int b = BLOCK_ROUND_DOWN(K);
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float A_shared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float B_shared[BLOCK_SIZE][BLOCK_SIZE];
    
    for(int i = 0; i < BLOCK_ROUND_DOWN(K); i+= BLOCK_SIZE) {

        A_shared[threadIdx.y][threadIdx.x] = A[K * y + i + threadIdx.x];        
        B_shared[threadIdx.y][threadIdx.x] = B[(i + threadIdx.y) * N + x];

        __syncthreads();

        #pragma unroll
        for(int j = 0; j < BLOCK_SIZE; ++j) {
            Cvalue += A_shared[threadIdx.y][j] * B_shared[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (a) {

        A_shared[threadIdx.y][threadIdx.x] = threadIdx.x < a ? A[K * y + b + threadIdx.x] : 0;        
        B_shared[threadIdx.y][threadIdx.x] = threadIdx.y < a ? B[(b + threadIdx.y) * N + x] : 0;

        __syncthreads();

        #pragma unroll
        for(int j = 0; j < BLOCK_SIZE; ++j) {
            Cvalue += A_shared[threadIdx.y][j] * B_shared[j][threadIdx.x];
        }

        __syncthreads();
    }

    if (y < M && x < N) {    
        C[y * N + x] = Cvalue;
    }
}

// __global__ void matMulCUDA(float *A, float *B, float *C, int M, int N, int K) {

//     float Cvalue = 0;

//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//     int x = blockIdx.x * blockDim.x + threadIdx.x;

//     if (y < M && x < N) {

//         for (int i = 0; i < K; ++i)
//             Cvalue += A[y * K + i] * B[i * N + x];

//         C[y * N + x] = Cvalue;
//     }
// }

void Matrix::matMul(Matrix const &A, Matrix const &B, Matrix &C) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(BLOCK_ROUND_UP(B.cols) / dimBlock.x, BLOCK_ROUND_UP(A.rows) / dimBlock.y);

    matMulCUDA<<<dimGrid, dimBlock>>>(A.buff, B.buff, C.buff, A.rows, B.cols, B.rows);
}

__global__ void matSumCUDA(float *A, float *B, float *C, int M, int N) {
    
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < M && x < N) {
        C[y * N + x] = A[y * N + x] + B[y * N + x];
    }
}

void Matrix::matSum(Matrix const &A, Matrix const &B, Matrix &C) {
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(BLOCK_ROUND_UP(B.cols) / dimBlock.x, BLOCK_ROUND_UP(A.rows) / dimBlock.y);

    matSumCUDA<<<dimGrid, dimBlock>>>(A.buff, B.buff, C.buff, A.rows, A.cols);
}


__global__ void matSubCUDA(float *A, float *B, float *C, int M, int N) {
    
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < M && x < N) {
        C[y * N + x] = A[y * N + x] - B[y * N + x];
    }
}

void Matrix::matSub(Matrix const &A, Matrix const &B, Matrix &C) {
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(BLOCK_ROUND_UP(B.cols) / dimBlock.x, BLOCK_ROUND_UP(A.rows) / dimBlock.y);

    matSubCUDA<<<dimGrid, dimBlock>>>(A.buff, B.buff, C.buff, A.rows, A.cols);
}

__global__ void matElMulCUDA(float *A, float *B, float *C, int M, int N) {
    
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < M && x < N) {
        C[y * N + x] = A[y * N + x] * B[y * N + x];
    }
}

void Matrix::matElMul(Matrix const &A, Matrix const &B, Matrix &C) {
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(BLOCK_ROUND_UP(B.cols) / dimBlock.x, BLOCK_ROUND_UP(A.rows) / dimBlock.y);

    matElMulCUDA<<<dimGrid, dimBlock>>>(A.buff, B.buff, C.buff, A.rows, A.cols);
}

__global__ void matScalarMulCUDA(float const e, float *A, float *B, int M, int N) {
    
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < M && x < N) {
        B[y * N + x] = e * A[y * N + x];
    }
}

void Matrix::matScalarMul(float const x, Matrix const &A, Matrix &B) {
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(BLOCK_ROUND_UP(B.cols) / dimBlock.x, BLOCK_ROUND_UP(A.rows) / dimBlock.y);

    matScalarMulCUDA<<<dimGrid, dimBlock>>>(x, A.buff, B.buff, A.rows, A.cols);
}

__global__ void matTCUDA(float *A, float *B, int M, int N) {
    
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < M && x < N) {
        B[x * M + y] = A[y * N + x];
    }
}

void Matrix::matT(Matrix const &A, Matrix &B) {

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(BLOCK_ROUND_UP(B.cols) / dimBlock.x, BLOCK_ROUND_UP(A.rows) / dimBlock.y);

    matTCUDA<<<dimGrid, dimBlock>>>(A.buff, B.buff, A.rows, A.cols);
}

__global__ void matReLUCUDA(float *A, float *B, int M, int N) {
    
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < M && x < N) {
        B[y * N + x] = A[y * N + x] > 0 ? A[y * N + x] : 0;
    }
}

void Matrix::matReLU(Matrix const &A, Matrix &B) {
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(BLOCK_ROUND_UP(B.cols) / dimBlock.x, BLOCK_ROUND_UP(A.rows) / dimBlock.y);

    matReLUCUDA<<<dimGrid, dimBlock>>>(A.buff, B.buff, A.rows, A.cols);
}




__global__ void matSigmoidCUDA(float *A, float *B, int M, int N) {
    
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < M && x < N) {
        B[y * N + x] = 1.0f / (1.0f + expf(-A[y * N + x]));
    }
}

void Matrix::matSigmoid(Matrix const &A, Matrix &B) {
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(BLOCK_ROUND_UP(B.cols) / dimBlock.x, BLOCK_ROUND_UP(A.rows) / dimBlock.y);

    matSigmoidCUDA<<<dimGrid, dimBlock>>>(A.buff, B.buff, A.rows, A.cols);
}

__global__ void matSigmoidPrimeCUDA(float *A, float *B, int M, int N) {
    
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < M && x < N) {
        float tmp = 1.0f / (1.0f + expf(-A[y * N + x]));
        B[y * N + x] = tmp * (1 - tmp);
    }
}

void Matrix::matSigmoidPrime(Matrix const &A, Matrix &B) {
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(BLOCK_ROUND_UP(B.cols) / dimBlock.x, BLOCK_ROUND_UP(A.rows) / dimBlock.y);

    matSigmoidPrimeCUDA<<<dimGrid, dimBlock>>>(A.buff, B.buff, A.rows, A.cols);
}










__global__ void matTanhCUDA(float *A, float *B, int M, int N) {
    
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < M && x < N) {
        B[y * N + x] = tanh(A[y * N + x]);
    }
}

void Matrix::matTanh(Matrix const &A, Matrix &B) {
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(BLOCK_ROUND_UP(B.cols) / dimBlock.x, BLOCK_ROUND_UP(A.rows) / dimBlock.y);

    matTanhCUDA<<<dimGrid, dimBlock>>>(A.buff, B.buff, A.rows, A.cols);
}

__global__ void matTanhPrimeCUDA(float *A, float *B, int M, int N) {
    
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < M && x < N) {
        float tmp = tanh(A[y * N + x]);
        B[y * N + x] = 1 - tmp;
    }
}

void Matrix::matTanhPrime(Matrix const &A, Matrix &B) {
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(BLOCK_ROUND_UP(B.cols) / dimBlock.x, BLOCK_ROUND_UP(A.rows) / dimBlock.y);

    matTanhPrimeCUDA<<<dimGrid, dimBlock>>>(A.buff, B.buff, A.rows, A.cols);
}

__global__ void matReLUPrimeCUDA(float *A, float *B, int M, int N) {
    
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < M && x < N) {
        B[y * N + x] = A[y * N + x] >= 0 ? 1 : 0;
    }
}

void Matrix::matReLUPrime(Matrix const &A, Matrix &B) {
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(BLOCK_ROUND_UP(B.cols) / dimBlock.x, BLOCK_ROUND_UP(A.rows) / dimBlock.y);

    matReLUPrimeCUDA<<<dimGrid, dimBlock>>>(A.buff, B.buff, A.rows, A.cols);
}

__global__ void matSoftmaxCUDA(float *A, float *B, int M, int N) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < N) {
        float m = A[x];

        for (int j = 0; j < M; ++j) {
            m = fmax(m, A[N * j + x]);
        }

        float sum = 0.0f;
        for (int j = 0; j < M; ++j) {
            sum += expf(A[N * j + x] - m);
        }

        for (int j = 0; j < M; ++j) {
            B[N * j + x] = expf(A[N * j + x] - m) / sum;
        }
    }
}

void Matrix::matColSoftmax(Matrix const &A, Matrix &B) {
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid(BLOCK_ROUND_UP(B.cols) / dimBlock.x);

    matSoftmaxCUDA<<<dimGrid, dimBlock>>>(A.buff, B.buff, A.rows, B.cols);
}


float Matrix::cost(Matrix const &A, Matrix const &B) {

    float *tmp_A = new float[A.size()];
    float *tmp_B = new float[B.size()];

    cudaMemcpy(tmp_A, A.buff, A.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(tmp_B, B.buff, B.size() * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0;
    for (int i = 0; i < A.cols; ++i) {
        for (int j = 0; j  < A.rows; ++j) {
            sum += logf(tmp_A[A.cols * j + i]) * (-tmp_B[B.cols * j + i]);
        }
    }

    delete tmp_A;
    delete tmp_B;

    return sum / A.cols;
}

void Matrix::initialize(Initializer &initializer) {
    float *tmp_buff = new float[this->size()];
    initializer.fill(tmp_buff, this->size());
    cudaMemcpy(this->buff, tmp_buff, this->size() * sizeof(float), cudaMemcpyHostToDevice);
}

void Matrix::initialize(float *buff) {
    cudaMemcpy(this->buff, buff, this->size() * sizeof(float), cudaMemcpyHostToDevice);
}

int Matrix::size() const {
    return this->rows * this->cols;
}
int Matrix::getRows() const {
    return this->rows;
}
int Matrix::getCols() const {
    return this->cols;
}

// Used only for testing
bool Matrix::operator==(Matrix const &other) const {

    if (this->rows != other.rows || this->cols != other.cols) {
        return false;
    }

    float *a = new float[this->rows * this->cols];
    float *b = new float[other.rows * other.cols];

    cudaMemcpy(a, this->buff, this->rows * this->cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, other.buff, other.rows * other.cols * sizeof(float), cudaMemcpyDeviceToHost);

    bool equal = true;
    for (int i = 0; i < this->rows * this->cols; ++i) {
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
    ss << std::fixed << std::setprecision(2);
    
    float *a = new float[this->rows * this->cols];

    cudaMemcpy(a, this->buff, this->rows * this->cols * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            
            if (j) {
                ss << " ";
            }
            
            ss << a[this->cols * i + j];
        }
        ss << std::endl;
    }

    delete a;

    return ss.str();
}

std::ostream& operator<<(std::ostream& stream, Matrix const &matrix) {
    return stream << "Matrix[" << matrix.rows << ", " << matrix.cols << "]";
}