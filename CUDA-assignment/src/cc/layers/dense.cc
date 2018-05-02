// #include "dense.hh"

// void Dense::update(float learning_rate) {
//     // std::cout << "Dense::update: weights" << std::endl;
//     // std::cout << this->weights.toString() << std::endl;

//     Matrix::matMul(*(this->d), this->inputT, this->dweights);
//     Matrix::matScalarMul(learning_rate, this->dweights, this->dweights);
//     Matrix::matSub(this->weights, this->dweights, this->weights);
//     Matrix::matT(this->weights, this->weightsT);

//     // std::cout << "Dense::update: dweights" << std::endl;
//     // std::cout << this->dweights.toString() << std::endl;
//     // std::cout << "Dense::update: weights" << std::endl;
//     // std::cout << this->weights.toString() << std::endl;
// };


// #include <sstream>

// #ifndef NDEBUG
//     const bool debug = true;
// #else
//     const bool debug = false;
// #endif

// Dense::Dense(Layer *prev, int size) : Dense(prev->size, size) {}

// Dense::Dense(int prev_size, int size) {

// }

// int Dense::build(int data_len, int prev_size) {

    // this->w = &Matrix(this->size, prev_size);
    // this->dw = new Matrix(this->size, prev_size);
    // this->wT = new Matrix(prev_size, this->size);
    // this->gT = new Matrix(data_len, prev_size);
    // this->d = new Matrix(prev_size, data_len);
    // this->g = new Matrix(this->size, data_len);

    // if (debug) std::cerr << "Dense::build: w=" << *(this->w) 
            // << ", dw=" << *(this->dw) << ", wT=" << *(this->wT) << ", gT=" << *(this->gT) 
            // << ", d=" << *(this->d) << ", g=" << *(this->g) << std::endl;

    // return this->size;
// }

// Matrix& Dense::forward_pass() {

    // if (this->prev != this) {
    //     matTranspose(&(this->prev->getg()), gT);
    //     matMul(this->w, &(this->prev->getg()), this->g);
    // }

    // return Layer::forward_pass();
// }

// void Dense::backward_pass(Matrix &output) {

    // matMul(wT, &output, d);

    // // Update weights
    // matMul(&output, gT, dw);
    // matScalarMul(dw, -1); // eta here
    // matSum(w, dw, w);
    // matTranspose(w, wT);

    // return Layer::backward_pass(*d);
// }


// std::string Dense::info() {
//     std::stringstream ss;
//     ss << "DENSE (" << this->size << ")";
//     return ss.str();
// }