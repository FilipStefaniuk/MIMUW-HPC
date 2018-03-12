#include <omp.h>
#include <iostream>
#include <iomanip>

#define STEPS 10000
#define THREADS 16 //you can also use the OMP_NUM_THREADS environmental variable

double powerParallelReduction(double, long);
double powerParallelCritical(double, long);

double power(double x, long n) {
    if (n == 0) {
        return 1;
    }

    return x * power(x, n - 1);
}

double calcPi(long n) {

    if (n < 0) {
        return 0;
    }

    return 1.0 / power(16, n)
           * (4.0 / (8 * n + 1.0)
              - 2.0 / (8 * n + 4.0)
              - 1.0 / (8 * n + 5.0)
              - 1.0/(8 * n + 6.0))
           + calcPi(n - 1);
}

double powerParallelReduction(double x, long n) {

    double total = 1;

    #pragma omp parallel for reduction (*:total)
    for (long i = 0; i < n; ++i) {
        total *= x;
    }

    return total;
}

double powerParallelCritical(double x, long n) {
    double total = 1;

    #pragma omp parallel
    {
        double tmpTotal = 1;
        #pragma omp for nowait
        for (long i = 0; i < n; ++i) {
            tmpTotal *= x;
        }

        #pragma omp critical
        {
            total *= tmpTotal;
        }
    }

    return total;
}

double calcPiParallelReduction(long n) {
    double total = 0;

    #pragma omp parallel for reduction (+:total)
    for (long i = 0; i <= n; ++i) {
            total += 1.0 / powerParallelCritical(16, i)
           * (4.0 / (8 * i + 1.0)
              - 2.0 / (8 * i + 4.0)
              - 1.0 / (8 * i + 5.0)
              - 1.0/(8 * i + 6.0));
    }

    return total;
}

double calcPiParallelCritical(long n) {
    double total = 0;

    #pragma omp parallel
    {
        double tmpTotal = 0;
        #pragma omp for nowait
        for (long i = 0; i <= n; ++i) {
            tmpTotal += 1.0 / powerParallelCritical(16, i)
              * (4.0 / (8 * i + 1.0)
              - 2.0 / (8 * i + 4.0)
              - 1.0 / (8 * i + 5.0)
              - 1.0/(8 * i + 6.0));        
        }

        #pragma omp critical 
        {
            total += tmpTotal;
        }
    }

    return total;
}

double calcWithCounter(long n) {
        double startTime = omp_get_wtime ();
        double result = calcPi(n);
        double endTime = omp_get_wtime();
        std::cout << "Sequential execution time: " <<  endTime - startTime << std::endl;
        return result;
}

double calcPiParallelReductionWithCounter(long n) {
        double startTime = omp_get_wtime ();
        double result = calcPiParallelReduction(n);
        double endTime = omp_get_wtime();
        std::cout << "Parallel Reduction execution time: " <<  endTime - startTime << std::endl;
        return result;
}

double calcPiParallelCriticalWithCounter(long n) {
        double startTime = omp_get_wtime ();
        double result = calcPiParallelCritical(n);
        double endTime = omp_get_wtime();
        std::cout << "Parallel Critical execution time: " <<  endTime - startTime << std::endl;
        return result;
}

int main(int argc, char *argv[]) {
    std::cout << std::setprecision(10) << calcWithCounter(STEPS) << std::endl;
    std::cout << std::setprecision(10) << calcPiParallelReductionWithCounter(STEPS) << std::endl;
    std::cout << std::setprecision(10) << calcPiParallelCriticalWithCounter(STEPS) << std::endl;
    return 0;
}
