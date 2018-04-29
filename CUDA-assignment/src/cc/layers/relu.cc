// #include "relu.hh"
// #include <sstream>

// #ifndef NDEBUG
//     const bool debug = true;
// #else
//     const bool debug = false;
// #endif

// int ReLU::build(int data_len, int prev_size) {

//     // this->g = new Matrix(prev_size, data_len);
//     // this->d = new Matrix(prev_size, data_len);

//     // if (debug) std::cerr << "ReLU::build: d=" << *(this->d) << ", g=" << *(this->g) << std::endl;

//     return prev_size;
// }

// Matrix& ReLU::forward_pass() {
// //     if (this->prev != this) {
// //         matRelu(&(this->prev->getg()), g);
// //     }
// //     return Layer::forward_pass();
// }

// void ReLU::backward_pass(Matrix &output) {
// //     matReluBack(g, g);
// //     matElMul(&output, g, d);

// //     Layer::backward_pass(*d);
// }

// std::string ReLU::info() {
//     return "ReLU";
// }