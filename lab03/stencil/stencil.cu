#include <time.h>
#include <stdio.h>

#define RADIUS        3000
#define NUM_ELEMENTS  1000

static void handleError(cudaError_t err, const char *file, int line ) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
#define cudaCheck( err ) (handleError(err, __FILE__, __LINE__ ))

__global__ void stencil_1d(int *in, int *out) {

  //PUT YOUR CODE HERE
  int tid = blockIdx.x;
  int lo = tid - RADIUS < 0 ? 0 : tid - RADIUS;
  int hi = tid + RADIUS > NUM_ELEMENTS ? NUM_ELEMENTS : tid + RADIUS;
  
  out[tid] = 0;
  for (int i = lo; i < hi; ++i) {
	out[tid] += in[i]; 
  }
}

void cpu_stencil_1d(int *in, int *out) {

  //PUT YOUR CODE HERE
  for (int j = 0; j < NUM_ELEMENTS; ++j) {
    int tid = j;
    int lo = tid - RADIUS < 0 ? 0 : tid - RADIUS;
    int hi = tid + RADIUS > NUM_ELEMENTS ? NUM_ELEMENTS : tid + RADIUS;
  
    out[tid] = 0;
    for (int i = lo; i < hi; ++i) {
          out[tid] += in[i];
    }  
  }
}

int main() {
  
  //PUT YOUR CODE HERE - INPUT AND OUTPUT ARRAYS
  int in[NUM_ELEMENTS], out[NUM_ELEMENTS];
  int *devIn, *devOut;  

  for (int i = 0; i < NUM_ELEMENTS; ++i) {
    in[i] = rand() % 1000;
  }

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord( start, 0 );

  //PUT YOUR CODE HERE - DEVICE MEMORY ALLOCATION
  cudaMalloc((void**)&devIn, NUM_ELEMENTS * sizeof(int));
  cudaMalloc((void**)&devOut, NUM_ELEMENTS * sizeof(int));

  cudaMemcpy(devIn, in, NUM_ELEMENTS * sizeof(int), cudaMemcpyHostToDevice);

  //PUT YOUR CODE HERE - KERNEL EXECUTION
  stencil_1d<<<NUM_ELEMENTS, 1>>>(devIn, devOut);  

  cudaCheck(cudaPeekAtLastError());

  //PUT YOUR CODE HERE - COPY RESULT FROM DEVICE TO HOST
  cudaMemcpy(out, devOut, NUM_ELEMENTS * sizeof(int), cudaMemcpyDeviceToHost);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsedTime;
  cudaEventElapsedTime( &elapsedTime, start, stop);
  printf("Total GPU execution time:  %3.1f ms\n", elapsedTime);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  //PUT YOUR CODE HERE - FREE DEVICE MEMORY  
  cudaFree(devIn);
  cudaFree(devOut);

  struct timespec cpu_start, cpu_stop;
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_start);

  cpu_stencil_1d(in, out);

  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &cpu_stop);
  double result = (cpu_stop.tv_sec - cpu_start.tv_sec) * 1e3 + (cpu_stop.tv_nsec - cpu_start.tv_nsec) / 1e6;
  printf( "CPU execution time:  %3.1f ms\n", result);
  
  return 0;
}


