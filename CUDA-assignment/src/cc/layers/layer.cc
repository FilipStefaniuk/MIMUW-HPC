#include "layer.hh"

#ifndef NDEBUG
    const bool debug = true;
#else
    const bool debug = false;
#endif

// Matrix& Layer::build(Matrix &input) {
//     if (this->next != this) {
//         return this->next->build(input);
//     }
//     return input;
// }

// Matrix& Layer::forward_pass() {
//     if (this->next != this) {
//         return this->next->forward_pass();
//     }

//     return *(this->g);
// }
        
// void Layer::backward_pass(Matrix &output) {
//     if (this->prev != this) {
//         this->backward_pass(output);
//     }
// } 

// Matrix& Layer::getg() {
//     return *g;
// }

// Matrix& Layer::getd() {
//     return *d;
// }
