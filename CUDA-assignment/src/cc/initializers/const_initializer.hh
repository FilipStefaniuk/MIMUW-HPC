#ifndef __CONST_INITIALIZER_HH__
#define __CONST_INITIALIZER_HH__

#include "initializer.hh"

class ConstInitializer : public Initializer {

    private:

        float val;

    public:
        ConstInitializer(float val) : val(val) {}

        void fill(float *buff, int size) {
            for (int i = 0; i < size; ++i, ++buff) {
                *buff = val;
            }
        }

};

#endif