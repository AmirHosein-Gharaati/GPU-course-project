
#include <stdlib.h>

extern "C"
{
#include "cuda_sort.h"
}

extern "C" void gpu_merge_sort(int *array, int size)
{
    int *gpuData;
    int *gpuAuxData;
    int left = 0;
    int right = size;

    cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);

    cudaMalloc((void **)&gpuData, size * sizeof(int));
    cudaMalloc((void **)&gpuAuxData, size * sizeof(int));
    cudaMemcpy(gpuData, array, size * sizeof(int), cudaMemcpyHostToDevice);

    simple_mergesort<<<1, 1>>>(gpuData, gpuAuxData, left, right, 0);
    cudaDeviceSynchronize();

    cudaMemcpy(array, gpuData, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(gpuAuxData);
    cudaFree(gpuData);

    cudaDeviceReset();
}