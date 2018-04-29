// #include <chrono>
#include "src/cc/matrix/matrix.hh"
#include "src/cc/timer.cuh"

const int Sizes[] = {10, 100, 500, 1000, 1500, 2000, 2500};

void fillRandom(float* buff, unsigned n) {
    for (int i = 0; i < n; ++i) {
        buff[i] = rand() / (float) RAND_MAX;
    }
}

int main() {

    // Test multiplication speed
    for (auto n : Sizes) {
        
        float *a = new float[n*n];
        float *b = new float[n*n];

        fillRandom(a, n*n);
        fillRandom(b, n*n);

        Matrix A(n, n, a);
        Matrix B(n, n, b);
        Matrix C(n, n);

        // auto start = std::chrono::steady_clock::now();
        wrapper(Matrix::matMul, A, B, C);
        // auto end = std::chrono::steady_clock::now();

        // std::cout << "Multiplication of square matrix of size " << n << ": " <<
        // std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() <<
        // " ms" << std::endl;

        delete a;
        delete b;
    }

}