#include "cuda_runtime.h"

void mergesort(long *data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid)