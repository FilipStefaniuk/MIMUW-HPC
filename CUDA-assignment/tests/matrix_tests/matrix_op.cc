#include <iostream>
#include <cassert>
#include "../../src/cc/matrix/matrix.hh"

#define OK "OK"

int main() {

    {
        std::cout << "MATRIX MULTIPLICATION" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {1., 3., 5., 7., 2., 4., 6., 8.}; 
        float buffB[] = {1., 8., 9., 2., 7., 10., 3., 6., 11., 4., 5., 12.};
        float buffC[] = {50., 94., 178., 60., 120., 220.};

        Matrix A(2, 4);
        Matrix B(4, 3);
        Matrix C(2, 3);
        Matrix D(2, 3);

        A.init(buffA);
        B.init(buffB);
        C.init(buffC);

        // std::cout << A.toString() << std::endl;
        // std::cout << B.toString() << std::endl;

        Matrix::matMul(A, B, D, 0);

        // std::cout << C.toString() << std::endl;
        // std::cout << D.toString() << std::endl;        

        assert(C == D);
        std::cout << OK << std::endl;       
    }

    {
        std::cout << "MATRIX T LEFT MULTIPLICATION" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {1., 2., 3., 4., 5., 6., 7., 8.,};
        float buffB[] = {1., 8., 9., 2., 7., 10., 3., 6., 11., 4., 5., 12.};
        float buffC[] = {50., 94., 178., 60., 120., 220.};

        Matrix A(4, 2);
        Matrix B(4, 3);
        Matrix C(2, 3);
        Matrix D(2, 3);

        A.init(buffA);
        B.init(buffB);
        C.init(buffC);

        // std::cout << A.toString() << std::endl;
        // std::cout << B.toString() << std::endl;

        Matrix::matMul(A, B, D, LEFT_T);

        // std::cout << C.toString() << std::endl;
        // std::cout << D.toString() << std::endl;        

        assert(C == D);
        std::cout << OK << std::endl;       
    }

    {
        std::cout << "MATRIX T RIGHT MULTIPLICATION" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {1., 3., 5., 7., 2., 4., 6., 8.}; 
        float buffB[] = {1., 2., 3., 4., 8., 7., 6., 5., 9., 10., 11., 12.};
        float buffC[] = {50., 94., 178., 60., 120., 220.};


        Matrix A(2, 4);
        Matrix B(3, 4);
        Matrix C(2, 3);
        Matrix D(2, 3);

        A.init(buffA);
        B.init(buffB);
        C.init(buffC);

        // std::cout << A.toString() << std::endl;
        // std::cout << B.toString() << std::endl;

        Matrix::matMul(A, B, D, RIGHT_T);

        // std::cout << C.toString() << std::endl;
        // std::cout << D.toString() << std::endl;        

        assert(C == D);
        std::cout << OK << std::endl;       
    }
    
    {
        std::cout << "MATRIX ROW SUM" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {1., 2., 3., 4., 
                         5., 6., 7., 8.}; 
        
        float buffB[] = {10., 
                         26.};
        
        Matrix A(2, 4);
        Matrix B(2, 1);
        Matrix C(2, 1);

        A.init(buffA);
        B.init(buffB);

        // std::cout << A.toString() << std::endl;
        // std::cout << B.toString() << std::endl;

        Matrix::rowSum(A, C);

        // std::cout << C.toString() << std::endl;

        assert(C == B);
        std::cout << OK << std::endl;       
    }

    {
        std::cout << "MATRIX VEC ADD" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {1., 2., 3., 4., 
                         5., 6., 7., 8.}; 
        
        float buffB[] = {20., 
                         10.};
        
        float buffC[] = {21., 22., 23., 24.,
                         15., 16., 17., 18.};

        Matrix A(2, 4);
        Matrix B(2, 1);
        Matrix C(2, 4);
        Matrix D(2, 4);

        A.init(buffA);
        B.init(buffB);
        C.init(buffC);

        // std::cout << A.toString() << std::endl;
        // std::cout << B.toString() << std::endl;

        Matrix::vecAdd(A, B, D);

        // std::cout << C.toString() << std::endl;
        // std::cout << D.toString() << std::endl;        

        assert(C == D);
        std::cout << OK << std::endl;       
    }

    {
        std::cout << "MATRIX SUB" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {1., 2., 3., 4., 5., 6., 7., 8.}; 
        float buffB[] = {8., 7., 6., 5., 4., 3., 2., 1.};
        float buffC[] = {-7., -5., -3., -1., 1., 3., 5., 7.};

        Matrix A(2, 4);
        Matrix B(2, 4);
        Matrix C(2, 4);
        Matrix D(2, 4);

        A.init(buffA);
        B.init(buffB);
        C.init(buffC);

        // std::cout << A.toString() << std::endl;
        // std::cout << B.toString() << std::endl;

        Matrix::matSub(A, B, D);

        // std::cout << C.toString() << std::endl;
        // std::cout << D.toString() << std::endl;        

        assert(C == D);
        std::cout << OK << std::endl;       
    }

    {
        std::cout << "MATRIX EL MUL" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {1., 2., 3., 4., 5., 6., 7., 8.}; 
        float buffB[] = {8., 7., 6., 5., 4., 3., 2., 1.};
        float buffC[] = {8., 14., 18., 20., 20., 18., 14., 8.};

        Matrix A(2, 4);
        Matrix B(2, 4);
        Matrix C(2, 4);
        Matrix D(2, 4);

        A.init(buffA);
        B.init(buffB);
        C.init(buffC);

        // std::cout << A.toString() << std::endl;
        // std::cout << B.toString() << std::endl;

        Matrix::matElMul(A, B, D);

        // std::cout << C.toString() << std::endl;
        // std::cout << D.toString() << std::endl;        

        assert(C == D);
        std::cout << OK << std::endl;       
    }

    {
        std::cout << "MATRIX SCALAR MUL" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {1., 2., 3., 4., 5., 6., 7., 8.}; 
        float buffB[] = {10., 20., 30., 40., 50., 60., 70., 80.};

        Matrix A(2, 4);
        Matrix B(2, 4);
        Matrix C(2, 4);

        A.init(buffA);
        B.init(buffB);

        // std::cout << A.toString() << std::endl;
        // std::cout << B.toString() << std::endl;

        Matrix::matElMul(10., A, C);

        // std::cout << C.toString() << std::endl;

        assert(B == C);
        std::cout << OK << std::endl;       
    }
}