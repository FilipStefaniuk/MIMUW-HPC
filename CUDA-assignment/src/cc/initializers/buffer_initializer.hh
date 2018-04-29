#ifndef __BUFFER_INITIALIZER_HH__
#define __BUFFER_INITIALIZER_HH__

#include <cstring>
#include "initializer.hh"

class BufferInitializer : public Initializer {

    private:

        float *buff;

    public:
        BufferInitializer(float *buff) : buff(buff) {}

        void fill(float *buff, int size) {
            std::memcpy(buff, this->buff, size * sizeof(float));
        }

};

#endif