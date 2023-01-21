
#include <stdlib.h>

extern "C"
{
#include "cuda_sort.h"
}

__global__ void simple_mergesort(int *data, int *dataAux, int begin, int end, int depth)
{
    int middle = (end + begin) / 2;
    int i0 = begin;
    int i1 = middle;
    int index;
    int n = end - begin;

    cudaStream_t s, s1;

    if (n < 2)
    {
        return;
    }

    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    simple_mergesort<<<1, 1, 0, s>>>(data, dataAux, begin, middle, depth + 1);
    cudaStreamDestroy(s);

    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    simple_mergesort<<<1, 1, 0, s1>>>(data, dataAux, middle, end, depth + 1);
    cudaStreamDestroy(s1);

    cudaDeviceSynchronize();

    for (index = begin; index < end; index++)
    {
        if (i0 < middle && (i1 >= end || data[i0] <= data[i1]))
        {
            dataAux[index] = data[i0];
            i0++;
        }
        else
        {
            dataAux[index] = data[i1];
            i1++;
        }
    }

    for (index = begin; index < end; index++)
    {
        data[index] = dataAux[index];
    }
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