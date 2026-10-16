#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

struct GpuTimer {
  cudaEvent_t a, b;
  GpuTimer() { CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b)); }
  ~GpuTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
  void start(cudaStream_t s = 0) { CK(cudaEventRecord(a, s)); }
  float stop(cudaStream_t s = 0) { CK(cudaEventRecord(b, s)); CK(cudaEventSynchronize(b)); float ms; CK(cudaEventElapsedTime(&ms, a, b)); return ms; }
};
