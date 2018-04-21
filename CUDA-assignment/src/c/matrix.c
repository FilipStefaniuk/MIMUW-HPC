#include "matrix.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

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

void matRelu(Matrix *A) {
    for (int i = 0; i < A->height; ++i) {
        for (int j = 0; j < A->width; ++j) {
            if (A->elements[A->width * i + j] < 0)
                A->elements[A->width * i + j] = 0.0f; 
        }
    }
}

void matReluBack(Matrix *A) {
    for (int i = 0; i < A->height; ++i) {
        for (int j = 0; j < A->width; ++j) {
            if (A->elements[A->width * i + j] < 0)
                A->elements[A->width * i + j] = 0.0f;
            else
                A->elements[A->width * i + j] = 1.0f;
        }
    }
}


void matSoftmax(Matrix *A) {
    for (int i = 0; i < A->height; ++i) {
        
        float m = A->elements[A->width * i];
        for (int j = 0; j < A->width; ++j) {
            if (A->elements[A->width * i + j] > m) {
                m = A->elements[A->width * i + j];
            }
        }

        float sum = 0.0f;
        for (int j = 0; j < A->width; ++j) {
            sum += expf(A->elements[A->width * i + j] - m);
        }

        for (int j = 0; j < A->width; ++j) {
            A->elements[A->width * i + j] 
                = expf(A->elements[A->width * i + j] - m - logf(sum));
        }
    }
}

void matSum(const Matrix *A, const Matrix *B, Matrix *C) {
    for (int i = 0; i < C->height; ++i) {
        for (int j = 0; j < C->width; ++j) {
            C->elements[C->width * i + j] = B->elements[B->width * i + j] + A->elements[A->width * i + j];
        }
    }
}

void matScalarMul(Matrix *A, float x) {
    for (int i = 0; i < A->height; ++i) {
        for (int j = 0; j < A->width; ++j) {
            A->elements[A->width * i + j] *= x;  
        }
    }
}

void matElMul(const Matrix *A,const Matrix *B, Matrix *C) {
    for (int i = 0; i < C->height; ++i) {
        for (int j = 0; j < C->width; ++j) {
            C->elements[C->width * i + j] = B->elements[B->width * i + j] * A->elements[A->width * i + j];
        }
    }
}

Matrix* genMatrix(int width, int height, float val) {
    Matrix *A = (Matrix*) malloc(sizeof(Matrix));
    A->width = width;
    A->height = height;
    A->elements = (float*) malloc(width * height * sizeof(float));
    for (int i = 0; i < width * height; ++i) {
        A->elements[i] = val;
    }
    
    return A;
}

Matrix* genRandomMatrix(int width, int height) {

    Matrix *A = genMatrix(width, height, 0);
    for (int i = 0; i < width * height; ++i) {
        A->elements[i] = rand() / (float) RAND_MAX;
    }

    return A;
}

Matrix* genTransposed(const Matrix * A) {
    Matrix *B = genMatrix(A->height, A->width, 0);

    for (int i = 0; i < B->height; ++i) {
        for (int j = 0; j < B->width; ++j) {
            B->elements[B->width * i + j] = A->elements[A->width * j + i];
        }
    }
    return B;
}

void freeMatrix(Matrix *A) {
    free(A->elements);
    free(A);
}

void matPrint(Matrix *A) {

    printf("(MATRIX: (%d, %d)\n", A->height, A->width);
    for (int i = 0; i < A->height; ++i) {
        for (int j = 0; j < A->width; ++j) {
            printf("%.1f ", A->elements[A->width * i + j]);  
        }
        printf("\n");
    }
    printf("\n");
}