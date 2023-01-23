#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

typedef struct mergeSortResult
{
    cudaError_t cudaStatus;
    char *msg;
} mergeSortResult_t;

__global__ void mergeSortKernel(int *arr, int *aux, unsigned int blockSize, const unsigned int last)
{
    int x = threadIdx.x;
    int start = blockSize * x;
    int end = start + blockSize - 1;
    int mid = start + (blockSize / 2) - 1;
    int l = start, r = mid + 1, i = start;

    if (end > last)
    {
        end = last;
    }
    if (start == end || end <= mid)
    {
        return;
    }

    while (l <= mid && r <= end)
    {
        if (arr[l] <= arr[r])
        {
            aux[i++] = arr[l++];
        }
        else
        {
            aux[i++] = arr[r++];
        }
    }

    while (l <= mid)
    {
        aux[i++] = arr[l++];
    }
    while (r <= end)
    {
        aux[i++] = arr[r++];
    }

    for (i = start; i <= end; i++)
    {
        arr[i] = aux[i];
    }
}

inline mergeSortResult_t mergeSortError(cudaError_t cudaStatus, char *msg)
{
    mergeSortResult_t error;
    error.cudaStatus = cudaStatus;
    error.msg = msg;
    return error;
}

inline mergeSortResult_t mergeSortSuccess()
{
    mergeSortResult_t success;
    success.cudaStatus = cudaSuccess;
    return success;
}

inline mergeSortResult_t doMergeSortWithCuda(int *arr, unsigned int size_arg, int *dev_arr, int *dev_aux)
{
    const unsigned int last = size - 1;
    const unsigned size = size_arg * sizeof(int);
    unsigned int threadCount;
    cudaError_t cudaStatus;
    char msg[1024];

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_arr, arr, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
        return mergeSortError(cudaStatus, "cudaMemcpy failed!");
    }

    for (unsigned int blockSize = 2; blockSize < 2 * size; blockSize *= 2)
    {
        threadCount = size / blockSize;
        if (size % blockSize > 0)
        {
            threadCount++;
        }

        // Launch a kernel on the GPU with one thread for each block.
        mergeSortKernel<<<1, threadCount>>>(dev_arr, dev_aux, blockSize, last);

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess)
        {
            sprintf(msg, "mergeSortKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            return mergeSortError(cudaStatus, msg);
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess)
        {
            sprintf(msg, "cudaDeviceSynchronize returned error code %d after launching mergeSortKernel!\n", cudaStatus);
            return mergeSortError(cudaStatus, msg);
        }
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(arr, dev_arr, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess)
    {
        return mergeSortError(cudaStatus, "cudaMemcpy failed!");
    }

    return mergeSortSuccess();
}

cudaError_t mergeSortWithCuda(int *arr, unsigned int size)
{
    const unsigned int size = size * sizeof(int);
    int *dev_arr = 0;
    int *dev_aux = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    // Allocate GPU buffers for two vectors (main and aux array).
    cudaStatus = cudaMalloc((void **)&dev_arr, size);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void **)&dev_aux, size);
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed!");
        cudaFree(dev_arr);
        return cudaStatus;
    }

    mergeSortResult_t result = doMergeSortWithCuda(arr, size, dev_arr, dev_aux);

    if (result.cudaStatus != cudaSuccess)
    {
        fprintf(stderr, result.msg);
    }

    cudaFree(dev_arr);
    cudaFree(dev_aux);

    return cudaStatus;
}