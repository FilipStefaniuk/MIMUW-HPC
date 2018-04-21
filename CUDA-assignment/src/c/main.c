#include <stdio.h>

#include "matrix.h"

int main() {

    // Matrix *B = genMatrix(5, 3, 4);
    // Matrix *A = genMatrix(5, 3, 2);
    // matSoftmax(A);
    // matElMul(A, B, A);
    Matrix *A = genRandomMatrix(5, 3);
    Matrix *B = genTransposed(A);

    for (int i = 0; i < A->height; ++i) {
        for(int j = 0; j < A->width; ++j) {
            printf("%f ", A->elements[i*A->width + j]);
        }
        printf("\n");
    }

    printf("\n");

    for (int i = 0; i < B->height; ++i) {
        for(int j = 0; j < B->width; ++j) {
            printf("%f ", B->elements[i*B->width + j]);
        }
        printf("\n");
    }

    return 0;
}