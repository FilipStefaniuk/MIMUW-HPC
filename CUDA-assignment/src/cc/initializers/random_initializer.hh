#ifndef __RANDOM_INITIALIZER_HH__
#define __RANDOM_INITIALIZER_HH__

#include <random>
#include "initializer.hh"

class RandomInitializer : public Initializer {

    std::mt19937 rng;
    std::uniform_real_distribution<float> distribution;

    public:
        RandomInitializer() : distribution(0.0f, 1.0f) {}

        void fill(float *buff, int size) {
            for (int i = 0; i < size; ++i, ++buff) {
                *buff = distribution(rng);
            }
        }

};

#endif