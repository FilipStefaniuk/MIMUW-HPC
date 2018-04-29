#ifndef __MATRIX_H_
#define __MATRIX_H_

typedef struct {
    int width;
    int height;
    float * elements;
} Matrix;

void matMul(const Matrix *A, const Matrix *B, Matrix *C);
void matSum(const Matrix *A, const Matrix *B, Matrix *C);
void matSub(const Matrix *A, const Matrix *B, Matrix *C);
void matElMul(const Matrix *A, const Matrix *B, Matrix *C);

void matScalarMul(Matrix *A, float x);
void matRelu(Matrix *A, Matrix *B);
void matReluBack(Matrix *A, Matrix *B);
void matSoftmax(Matrix *A, Matrix *B);
void matTranspose(const Matrix *A, Matrix *B);

Matrix* matArgMax(Matrix *A);
void matPrint(Matrix *A);
void matPrintSize(Matrix *A);

Matrix* genMatrix(int height, int width, float val);
Matrix* genRandomMatrix(int height, int width);

void freeMatrix(Matrix *A);

#endif