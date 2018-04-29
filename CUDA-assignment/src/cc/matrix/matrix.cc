#include <cmath>
#include <sstream>
#include <cstring>
#include <iomanip>
#include "matrix.hh"

Matrix::Matrix(int rows, int cols) 
    : rows(rows), cols(cols), buff(new float[rows * cols]) {}

// Matrix::Matrix(int rows, int cols, float *buff) 
//     : Matrix(rows, cols) {
//         std::memcpy(this->buff, buff, rows*cols * sizeof(float));
// }
    
Matrix::~Matrix() {
    delete buff;
}

void Matrix::matMul(Matrix const &A, Matrix const &B, Matrix &C) {
    
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            C.buff[B.cols * i + j] = 0;
            for (int k = 0; k < A.cols; ++k) {
                C.buff[B.cols * i + j] += 
                    A.buff[A.cols * i + k] * B.buff[B.cols * k + j]; 
            }
        }
    }
}

void Matrix::matSum(Matrix const &A, Matrix const &B, Matrix &C) {
   
   for (int i = 0; i < C.rows; ++i) {
        for (int j = 0; j < C.cols; ++j) {
            C.buff[C.cols * i + j] = 
                B.buff[B.cols * i + j] + A.buff[A.cols * i + j];
        }
    }
}

void Matrix::matSub(Matrix const &A, Matrix const &B, Matrix &C) {

   for (int i = 0; i < C.rows; ++i) {
        for (int j = 0; j < C.cols; ++j) {
            C.buff[C.cols * i + j] = 
                A.buff[A.cols * i + j] - B.buff[B.cols * i + j];
        }
    }
}

void Matrix::matElMul(Matrix const &A, Matrix const &B, Matrix &C) {

    for (int i = 0; i < C.rows; ++i) {
        for (int j = 0; j < C.cols; ++j) {
            C.buff[C.cols * i + j] = 
                B.buff[B.cols * i + j] * A.buff[A.cols * i + j];
        }
    }
}

void Matrix::matScalarMul(float const x, Matrix const &A, Matrix &B) {
    
    for (int i = 0; i < B.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            B.buff[B.cols * i + j] = x * A.buff[A.cols * i + j];
        }
    }
}

void Matrix::matT(Matrix const &A, Matrix &B) {
    
    for (int i = 0; i < B.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            B.buff[B.cols * i + j] = A.buff[A.cols * j + i];
        }
    }
}

void Matrix::matReLU(Matrix const &A, Matrix &B) {
    
    for (int i = 0; i < B.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            B.buff[B.cols * i + j] = 
                A.buff[A.cols * i + j] > 0 ? A.buff[A.cols * i + j] : 0;
        }
    }
}

void Matrix::matTanh(Matrix const &A, Matrix &B) {
    
    for (int i = 0; i < B.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            B.buff[B.cols * i + j] = tanhf(A.buff[A.cols * i + j]);
        }
    }
}

void Matrix::matSigmoid(Matrix const &A, Matrix &B) {
    
    for (int i = 0; i < B.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            B.buff[B.cols * i + j] = 1.0f / (1.0f + expf(-A.buff[A.cols * i + j]));
        }
    }
}

void Matrix::matSigmoidPrime(Matrix const &A, Matrix &B) {
    
    for (int i = 0; i < B.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            float tmp = 1.0f / (1.0f + expf(-A.buff[A.cols * i + j]));
            B.buff[B.cols * i + j] = tmp * (1 - tmp);
        }
    }
}

void Matrix::matTanhPrime(Matrix const &A, Matrix &B) {
    
    for (int i = 0; i < B.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            float tmp = tanhf(A.buff[A.cols * i + j]);
            B.buff[B.cols * i + j] = 1 - tmp * tmp;
        }
    }
}


void Matrix::matReLUPrime(Matrix const &A, Matrix &B) {
    
    for (int i = 0; i < B.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            B.buff[B.cols * i + j] = 
                A.buff[A.cols * i + j] >= 0 ? 1 : 0;
        }
    }
}

void Matrix::matColSoftmax(Matrix const &A, Matrix &B) {
    
    for (int i = 0; i < A.cols; ++i) {
        
        float m = A.buff[i];

        for (int j = 0; j < A.rows; ++j) {
            m = fmax(m, A.buff[A.cols * j + i]);
        }

        float sum = 0.0f;
        for (int j = 0; j < A.rows; ++j) {
            sum += expf(A.buff[A.cols * j + i] - m);
            // sum += expf(A.buff[A.cols * j + i]);
        }

        for (int j = 0; j < A.rows; ++j) {
            // B.buff[B.cols * j + i] = expf(A.buff[A.cols * j + i] - m -logf(sum));
            B.buff[B.cols * j + i] = expf(A.buff[A.cols * j + i] - m) / sum;
            // B.buff[B.cols * j + i] = expf(A.buff[A.cols * j + i]) / sum; 
        }
    }
}

float Matrix::cost(Matrix const &A, Matrix const &B) {
    float sum = 0;
    // std::cout << "COST VALUES" << std::endl;
    for (int i = 0; i < A.cols; ++i) {
        for (int j = 0; j  < A.rows; ++j) {
            float tmp = B.buff[B.cols * j + i] * -logf(A.buff[A.cols * j + i]);
            // std::cout << A.buff[A.cols * j + i] << " " << logf(0.001 + A.buff[A.cols * j + i]) << " " << (-B.buff[B.cols * j + i]) << " " << tmp << std::endl;
            sum += tmp;
        }
    }

    return sum / A.cols;
}

bool Matrix::operator==(Matrix const &other) const {
    
    if (this->rows != other.rows || this->cols != other.cols) {
        return false;
    }

    for (int i = 0; i < this->rows * this->cols; ++i) {
        if (this->buff[i] != other.buff[i]) {
            return false;
        }
    }

    return true;
}

std::string Matrix::toString() const {
    
    std::stringstream ss;

    ss << std::fixed << std::setprecision(2);
    
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            
            if (j) {
                ss << " ";
            }
            
            ss << this->buff[this->cols * i + j];
        }
        ss << std::endl;
    }

    return ss.str();
}

int Matrix::size() const {
    return this->cols * this->rows;
}

int Matrix::getCols() const {
    return this->cols;
}

int Matrix::getRows() const {
    return this->rows;
}

void Matrix::initialize(Initializer &initializer) {
    initializer.fill(this->buff, this->size());
}

void Matrix::initialize(float *buff) {
    std::memcpy(this->buff, buff, this->rows * this->cols * sizeof(float));
}

std::ostream& operator<<(std::ostream& stream, Matrix const &matrix) {
    return stream << "Matrix[" << matrix.rows << ", " << matrix.cols << "]";
}
