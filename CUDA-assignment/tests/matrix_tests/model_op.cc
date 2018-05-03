#include <iostream>
#include <cassert>
#include "../../src/cc/matrix/matrix.hh"
#include "../../src/cc/model/model.hh"

#define OK "OK"

int main() {
        
    {
        std::cout << "ACCURACY" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {0.1, 0.1, 0.1, 0.25, 
                         0.2, 0.1, 0.1, 0.25,
                         0.3, 0.7, 0.2, 0.25,
                         0.4, 0.1, 0.6, 0.25}; 
        
        float buffB[] = {0., 1., 0., 1., 
                         0., 0., 0., 0.,
                         0., 0., 0., 0.,
                         1., 0., 1., 0.};

        Matrix A(4, 4);
        Matrix B(4, 4);

        A.init(buffA);
        B.init(buffB);

        // std::cout << A.toString() << std::endl;
        // std::cout << B.toString() << std::endl;

        float acc = Model::accuracy(A, B);

        // std::cout << acc << std::endl;

        assert(acc == 0.75);
        std::cout << OK << std::endl;
    }

    {
        std::cout << "CROSS ENTROPY COST" << std::endl;
        std::cout << "---------------------" << std::endl;

        float buffA[] = {0.1, 0.1, 0.1, 0.25, 
                         0.2, 0.1, 0.1, 0.25,
                         0.3, 0.7, 0.2, 0.25,
                         0.4, 0.1, 0.6, 0.25}; 
        
        float buffB[] = {0., 1., 0., 1., 
                         0., 0., 0., 0.,
                         0., 0., 0., 0.,
                         1., 0., 1., 0.};

        Matrix A(4, 4);
        Matrix B(4, 4);

        A.init(buffA);
        B.init(buffB);

        // std::cout << A.toString() << std::endl;
        // std::cout << B.toString() << std::endl;

        float cost = Model::crossEntropyCost(A, B);

        std::cout << cost << std::endl;
    }

    return 0;
}