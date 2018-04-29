#include <iostream>
#include <utility>

// template <class F, class... ARGS>
// auto wrapper(F func, ARGS&&... args) {

//     auto r = func(std::forward<ARGS>(args)...);
//     std::cout << "it works" << std::endl;
//     return r;
// }

// int f(int x) {
//     std::cout << "Function call " << x<< std::endl;
// }


class A {
    int x;
    public:
    A(int x) {
        this->x = x;
    }
    void hello() {
        std::cout << "Hello" << this->x << std::endl;
    }    
};

template <typename T, typename... ARGS>
void foo(ARGS&&... args) {
    T a(10, std::forward<ARGS>(args)...);
    a.hello();
}


int main() {

    foo<A>();
    return 0;
}