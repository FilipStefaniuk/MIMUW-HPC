#ifndef __INITIALIZER_HH__
#define __INITIALIZER_HH__

class Initializer {
    public:
        virtual ~Initializer() {}
        virtual void fill(float *buff, int size)=0;
};

#endif