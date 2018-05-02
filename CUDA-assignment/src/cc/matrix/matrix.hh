#ifndef __MATRIX_HH__
#define __MATRIX_HH__

#include <iostream>
#include <string>

#define BLOCK_SIZE 32
#define BLOCK_ROUND_UP(x) ((x + BLOCK_SIZE-1) & (~(BLOCK_SIZE-1))) 

#define LEFT_T 1
#define RIGHT_T 2

class Matrix {

    private:
    
        int rows, cols;

    public:

        float *buff;
        
        Matrix(int rows, int cols);
        ~Matrix();

        int size() const { return this->rows * this->cols; }
        int getRows() const { return this->rows; }
        int getCols() const { return this->cols; }

        void init();
        void init(float val);
        void init(float *buff);

        static void matMul(Matrix const &A, Matrix const &B, Matrix &C, int mode);

        static void matSub(Matrix const &A, Matrix const &B, Matrix &C);
        
        static void matElMul(Matrix const &A, Matrix const &B, Matrix &C);
        static void matElMul(float const x, Matrix const &A, Matrix &B);
        
        bool operator==(Matrix const &other) const;

        std::string toString() const;
        friend std::ostream& operator<<(std::ostream& stream, Matrix const &matrix);
};


#endif