#include <iostream>
#include <cassert>
#include "src/cc/matrix/matrix.hh"

#define OK "OK"

int main() {

    {
        std::cout << "MATRIX MULTIPLICATION" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {1., 3., 5., 7., 2., 4., 6., 8.}; 
        float buffB[] = {1., 8., 9., 0., 2., 7., 10., 0., 3., 6., 11., 0., 4., 5., 12., 0.};
        float buffC[] = {50., 94., 178., 0., 60., 120., 220., 0.};

        Matrix A(2, 4);
        Matrix B(4, 4);
        Matrix C(2, 4);
        Matrix D(2, 4);

        A.initialize(buffA);
        B.initialize(buffB);
        C.initialize(buffC);

        // std::cout << A.toString() << std::endl;
        // std::cout << B.toString() << std::endl;

        Matrix::matMul(A, B, D);

        // std::cout << C.toString() << std::endl;
        // std::cout << D.toString() << std::endl;        

        assert(C == D);
        std::cout << OK << std::endl;       
    }

    {
        std::cout << "MATRIX T LEFT MULTIPLICATION" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {1., 2., 3., 4., 5., 6., 7., 8.,};
        float buffB[] = {1., 8., 9., 0., 2., 7., 10., 0., 3., 6., 11., 0., 4., 5., 12., 0.};
        float buffC[] = {50., 94., 178., 0., 60., 120., 220., 0.};

        Matrix A(4, 2);
        Matrix B(4, 4);
        Matrix C(2, 4);
        Matrix D(2, 4);

        A.initialize(buffA);
        B.initialize(buffB);
        C.initialize(buffC);

        std::cout << A.toString() << std::endl;
        std::cout << B.toString() << std::endl;

        Matrix::matMulT0(A, B, D);

        std::cout << C.toString() << std::endl;
        std::cout << D.toString() << std::endl;        

        assert(C == D);
        std::cout << OK << std::endl;       
    }

    {
        std::cout << "MATRIX T RIGHT MULTIPLICATION" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {1., 3., 5., 7., 2., 4., 6., 8.}; 
        float buffB[] = {1., 2., 3., 4., 8., 7., 6., 5., 9., 10., 11., 12., 0., 0., 0., 0};
        float buffC[] = {50., 94., 178., 0., 60., 120., 220., 0.};


        Matrix A(4, 2);
        Matrix B(4, 4);
        Matrix C(2, 4);
        Matrix D(2, 4);

        A.initialize(buffA);
        B.initialize(buffB);
        C.initialize(buffC);

        std::cout << A.toString() << std::endl;
        std::cout << B.toString() << std::endl;

        Matrix::matMulT1(A, B, D);

        std::cout << C.toString() << std::endl;
        std::cout << D.toString() << std::endl;        

        assert(C == D);
        std::cout << OK << std::endl;       
    }

    {
        std::cout << "MATRIX SUM" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {1., 2., 3., 4., 5., 6., 7., 8.}; 
        float buffB[] = {8., 7., 6., 5., 4., 3., 2., 1.};
        // float buffC[] = {50., 94., 178., 60., 120., 220.};

        Matrix A(2, 4);
        Matrix B(2, 4);
        // Matrix C(2, 3);
        Matrix D(2, 4);

        A.initialize(buffA);
        B.initialize(buffB);
        // C.initialize(buffC);

        std::cout << A.toString() << std::endl;
        std::cout << B.toString() << std::endl;

        Matrix::matSum(A, B, D);

        // std::cout << C.toString() << std::endl;
        std::cout << D.toString() << std::endl;        

        // assert(C == D);
        // std::cout << OK << std::endl;       
    }

    {
        std::cout << "MATRIX SUB" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {1., 2., 3., 4., 5., 6., 7., 8.}; 
        float buffB[] = {8., 7., 6., 5., 4., 3., 2., 1.};
        // float buffC[] = {50., 94., 178., 60., 120., 220.};

        Matrix A(2, 4);
        Matrix B(2, 4);
        // Matrix C(2, 3);
        Matrix D(2, 4);

        A.initialize(buffA);
        B.initialize(buffB);
        // C.initialize(buffC);

        std::cout << A.toString() << std::endl;
        std::cout << B.toString() << std::endl;

        Matrix::matSub(A, B, D);

        // std::cout << C.toString() << std::endl;
        std::cout << D.toString() << std::endl;        

        // assert(C == D);
        // std::cout << OK << std::endl;       
    }

    {
        std::cout << "MATRIX EL MUL" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {1., 2., 3., 4., 5., 6., 7., 8.}; 
        float buffB[] = {8., 7., 6., 5., 4., 3., 2., 1.};
        // float buffC[] = {50., 94., 178., 60., 120., 220.};

        Matrix A(2, 4);
        Matrix B(2, 4);
        // Matrix C(2, 3);
        Matrix D(2, 4);

        A.initialize(buffA);
        B.initialize(buffB);
        // C.initialize(buffC);

        std::cout << A.toString() << std::endl;
        std::cout << B.toString() << std::endl;

        Matrix::matElMul(A, B, D);

        // std::cout << C.toString() << std::endl;
        std::cout << D.toString() << std::endl;        

        // assert(C == D);
        // std::cout << OK << std::endl;       
    }

    {
        std::cout << "MATRIX SCALAR MUL" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {1., 2., 3., 4., 5., 6., 7., 8.}; 
        // float buffC[] = {50., 94., 178., 60., 120., 220.};

        Matrix A(2, 4);
        // Matrix B(2, 4);
        Matrix C(2, 4);

        A.initialize(buffA);
        // C.initialize(buffC);

        std::cout << A.toString() << std::endl;
        // std::cout << B.toString() << std::endl;

        Matrix::matScalarMul(10., A, C);

        std::cout << C.toString() << std::endl;

        // assert(C == D);
        // std::cout << OK << std::endl;       
    }

    {
        std::cout << "MATRIX TRANSPOSITION" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {1., 2., 3., 4., 5., 6., 7., 8.}; 
        float buffB[] = {1., 5., 2., 6., 3., 7., 4., 8.};

        Matrix A(2, 4);
        Matrix B(4, 2);
        Matrix C(4, 2);

        A.initialize(buffA);
        B.initialize(buffB);

        std::cout << A.toString() << std::endl;

        Matrix::matT(A, C);

        std::cout << B.toString() << std::endl;
        std::cout << C.toString() << std::endl;

        // assert(B == C);
        // std::cout << OK << std::endl;
    }

    {
        std::cout << "MATRIX ReLU" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {1., 2., 3., 4., 5., 6., 7., 8., 9.}; 
        // float buffB[] = {1., 5., 2., 6., 3., 7., 4., 8.};

        Matrix A(3, 3);
        Matrix B(3, 3);
        Matrix C(3, 3);

        A.initialize(buffA);
        // B.initialize(buffB);

        std::cout << A.toString() << std::endl;

        Matrix::matReLU(A, C);

        // std::cout << B.toString() << std::endl;
        std::cout << C.toString() << std::endl;

        // assert(B == C);
        // std::cout << OK << std::endl;
    }

    {
        std::cout << "MATRIX ReLUPrime" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {-4, -3., -2., -1., 0., 1., 2., 3., 4.}; 
        // float buffB[] = {1., 5., 2., 6., 3., 7., 4., 8.};

        Matrix A(3, 3);
        Matrix B(3, 3);
        Matrix C(3, 3);

        A.initialize(buffA);
        // B.initialize(buffB);

        std::cout << A.toString() << std::endl;

        Matrix::matReLUPrime(A, C);

        // std::cout << B.toString() << std::endl;
        std::cout << C.toString() << std::endl;

        // assert(B == C);
        // std::cout << OK << std::endl;
    }

    {
        std::cout << "MATRIX Tanh" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {0.4, 0.3, 0.2, 0.1, 0., -0.1, -0.2, -0.3, -0.4}; 
        // float buffB[] = {1., 5., 2., 6., 3., 7., 4., 8.};

        Matrix A(3, 3);
        Matrix B(3, 3);
        Matrix C(3, 3);

        A.initialize(buffA);
        // B.initialize(buffB);

        std::cout << A.toString() << std::endl;

        Matrix::matTanh(A, C);

        // std::cout << B.toString() << std::endl;
        std::cout << C.toString() << std::endl;

        // assert(B == C);
        // std::cout << OK << std::endl;
    }

    {
        std::cout << "MATRIX TanhPrime" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {0.4, 0.3, 0.2, 0.1, 0., -0.1, -0.2, -0.3, -0.4}; 
        // float buffB[] = {1., 5., 2., 6., 3., 7., 4., 8.};

        Matrix A(3, 3);
        Matrix B(3, 3);
        Matrix C(3, 3);

        A.initialize(buffA);
        // B.initialize(buffB);

        std::cout << A.toString() << std::endl;

        Matrix::matTanhPrime(A, C);

        // std::cout << B.toString() << std::endl;
        std::cout << C.toString() << std::endl;

        // assert(B == C);
        // std::cout << OK << std::endl;
    }

    {
        std::cout << "MATRIX Sigmoid" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {0.4, 0.3, 0.2, 0.1, 0., -0.1, -0.2, -0.3, -0.4}; 
        // float buffB[] = {1., 5., 2., 6., 3., 7., 4., 8.};

        Matrix A(3, 3);
        Matrix B(3, 3);
        Matrix C(3, 3);

        A.initialize(buffA);
        // B.initialize(buffB);

        std::cout << A.toString() << std::endl;

        Matrix::matSigmoid(A, C);

        // std::cout << B.toString() << std::endl;
        std::cout << C.toString() << std::endl;

        // assert(B == C);
        // std::cout << OK << std::endl;
    }


    {
        std::cout << "MATRIX Sigmoid Prime" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {0.4, 0.3, 0.2, 0.1, 0., -0.1, -0.2, -0.3, -0.4}; 
        // float buffB[] = {1., 5., 2., 6., 3., 7., 4., 8.};

        Matrix A(3, 3);
        Matrix B(3, 3);
        Matrix C(3, 3);

        A.initialize(buffA);
        // B.initialize(buffB);

        std::cout << A.toString() << std::endl;

        Matrix::matSigmoidPrime(A, C);

        // std::cout << B.toString() << std::endl;
        std::cout << C.toString() << std::endl;

        // assert(B == C);
        // std::cout << OK << std::endl;
    }
    
    {
        std::cout << "MATRIX COL SOFTMAX" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {10., 1., 1., 5., 1., 2., 5., 1., 3.}; 
        // float buffB[] = {1., 5., 2., 6., 3., 7., 4., 8.};

        Matrix A(3, 3);
        Matrix B(3, 3);
        Matrix C(3, 3);

        A.initialize(buffA);
        // B.initialize(buffB);

        std::cout << A.toString() << std::endl;

        Matrix::matColSoftmax(A, C);

        // std::cout << B.toString() << std::endl;
        std::cout << C.toString() << std::endl;

        // assert(B == C);
        // std::cout << OK << std::endl;
    }

    return 0;
}