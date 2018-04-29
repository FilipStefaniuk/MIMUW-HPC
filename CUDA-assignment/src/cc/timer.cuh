#ifndef __TIMER_CUH__
#define __TIMER_CUH__

#include <iostream>

template <typename F, typename ...Args>
void wrapper(F func, Args&&... args) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord( start, 0 );

    func(std::forward<Args>(args)...);

    cudaEventRecord( start, 0 );    
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime( &elapsedTime, start, stop);
    std::cout << "Time: " << elapsedTime << " ms" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

#endif