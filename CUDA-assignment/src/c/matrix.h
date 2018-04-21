#ifndef __MATRIX_H_
#define __MATRIX_H_

typedef struct {
    int width;
    int height;
    float * elements;
} Matrix;

void matMul(const Matrix *A, const Matrix *B, Matrix *C);
void matSum(const Matrix *A, const Matrix *B, Matrix *C);
void matElMul(const Matrix *A, const Matrix *B, Matrix *C);

void matScalarMul(Matrix *A, float x);
void matRelu(Matrix *A);
void matReluBack(Matrix *A);
void matSoftmax(Matrix *A);

Matrix* matArgMax(Matrix *A);
void matPrint(Matrix *A);

Matrix* genMatrix(int width, int height, float val);
Matrix* genRandomMatrix(int width, int height);
Matrix* genTransposed(const Matrix *A);

void freeMatrix(Matrix *A);

#endif