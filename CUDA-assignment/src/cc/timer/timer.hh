#ifndef __TIMER_HH__
#define __TIMER_HH__

#include <chrono>
#include <iostream>

template <typename F, typename ...Args>
void wrapper(F func, Args&&... args) {

    auto start = std::chrono::steady_clock::now();
    func(std::forward<Args>(args)...);
    auto end = std::chrono::steady_clock::now();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Time: " << milliseconds << " ms" << std::endl;
}

#endif
