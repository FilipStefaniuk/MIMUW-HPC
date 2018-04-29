#ifndef __MATRIX_HH__
#define __MATRIX_HH__

#include "../initializers/initializer.hh"
#include <iostream>
#include <string>

class Matrix {

    private:
        int rows, cols;
        float *buff;

    public:
        Matrix(int rows, int cols);
        // Matrix(int rows, int cols, float *buff);
        ~Matrix();

        static void matMul(Matrix const &A, Matrix const &B, Matrix &C);
        static void matSum(Matrix const &A, Matrix const &B, Matrix &C);
        static void matSub(Matrix const &A, Matrix const &B, Matrix &C);
        static void matElMul(Matrix const &A, Matrix const &B, Matrix &C);
        static void matScalarMul(float const x, Matrix const &A, Matrix &B);
        static void matT(Matrix const &A, Matrix &B);
        
        static void matReLU(Matrix const &A, Matrix &B);
        static void matTanh(Matrix const &A, Matrix &B);
        static void matSigmoid(Matrix const &A, Matrix &B);
        static void matSigmoidPrime(Matrix const &A, Matrix &B);
        static void matTanhPrime(Matrix const &A, Matrix &B);
        static void matReLUPrime(Matrix const &A, Matrix &B);
        static void matColSoftmax(Matrix const &A, Matrix &B);

        static float cost(Matrix const &A, Matrix const &B);

        void initialize(Initializer &initializer);
        void initialize(float *buff);

        int size() const;
        int getRows() const;
        int getCols() const;

        bool operator==(Matrix const &other) const;

        std::string toString() const;
        friend std::ostream& operator<<(std::ostream& stream, Matrix const &matrix);
};


#endif