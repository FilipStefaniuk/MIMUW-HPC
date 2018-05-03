#include <cmath>
#include <sstream>
#include <cstring>
#include <iomanip>
#include <random>
#include "matrix.hh"

Matrix::Matrix(int rows, int cols) 
    : rows(rows), cols(cols), buff(new float[rows * cols]) {}
    
Matrix::~Matrix() {
    delete buff;
}

std::ostream& operator<<(std::ostream& stream, Matrix const &matrix) {
    return stream << "Matrix[" << matrix.rows << ", " << matrix.cols << "]";
}

//-----------------------------------------------------------------------------
//                            MATRIX MULTIPLICATION                                       
//-----------------------------------------------------------------------------

void Matrix::matMul(Matrix const &A, Matrix const &B, Matrix &C, int mode) {
    if (mode == LEFT_T) {
        for (int i = 0; i < A.cols; ++i) {
            for (int j = 0; j < B.cols; ++j) {
                C.buff[B.cols * i + j] = 0;
                for (int k = 0; k < A.rows; ++k) {
                    C.buff[B.cols * i + j] += 
                        A.buff[A.cols * k + i] * B.buff[B.cols * k + j]; 
                }
            }
        }
    } else if (mode == RIGHT_T) {
        for (int i = 0; i < C.rows; ++i) {
            for (int j = 0; j < C.cols; ++j) {
                C.buff[C.cols * i + j] = 0;
                for (int k = 0; k < A.cols; ++k) {
                    C.buff[C.cols * i + j] += 
                        A.buff[A.cols * i + k] * B.buff[B.cols * j + k]; 
                }
            }
        }
    } else {
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
}

//-----------------------------------------------------------------------------
//                            MATRIX ROW SUM                                      
//-----------------------------------------------------------------------------

void Matrix::rowSum(Matrix const &A, Matrix &B) {
    
    for (int i = 0; i < A.rows; ++i) {
        B.buff[i] = 0.0f;
        for (int j = 0; j < A.cols; ++j) {
            B.buff[i] += A.buff[i * A.cols + j];
        }
    }
}


//-----------------------------------------------------------------------------
//                            MATRIX ADD VECTOR                                      
//-----------------------------------------------------------------------------

void Matrix::vecAdd(Matrix const &A, Matrix const &B, Matrix &C) {
    
    for (int i = 0; i < C.rows; ++i) {
        for (int j = 0; j < C.cols; ++j) {
            C.buff[i * C.cols + j] = A.buff[i * C.cols + j] + B.buff[i];
        }
    }
}

//-----------------------------------------------------------------------------
//                            MATRIX SUBTRACTION                                       
//-----------------------------------------------------------------------------

void Matrix::matSub(Matrix const &A, Matrix const &B, Matrix &C) {

    for (int i = 0; i < C.rows * C.cols; ++i) {
        C.buff[i] = A.buff[i] - B.buff[i];
    }
}

//-----------------------------------------------------------------------------
//                            MATRIX EL MUL                                       
//-----------------------------------------------------------------------------

void Matrix::matElMul(Matrix const &A, Matrix const &B, Matrix &C) {

    for (int i = 0; i < C.rows * C.cols; ++i) {
        C.buff[i] = A.buff[i] * B.buff[i];
    }
}

void Matrix::matElMul(float const x, Matrix const &A, Matrix &B) {
    
    for (int i = 0; i < B.rows * B.cols; ++i) {
        B.buff[i] = x * A.buff[i];
    }
}

//-----------------------------------------------------------------------------
//                            INIT                                       
//-----------------------------------------------------------------------------

void Matrix::init() {

    std::mt19937 rng;
    std::uniform_real_distribution<float> distribution(0.0f, 1.0);

    for (int i = 0; i < this->rows * this->cols; ++i) {
        this->buff[i] = distribution(rng);
    }
}

void Matrix::init(float val) {
    for (int i = 0; i < this->rows * this->cols; ++i) {
        this->buff[i] = val;
    }
}

void Matrix::init(float *buff) {
    memcpy(this->buff, buff, this->cols * this->rows * sizeof(float));
}

//-----------------------------------------------------------------------------
//                                                                   
//-----------------------------------------------------------------------------

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
