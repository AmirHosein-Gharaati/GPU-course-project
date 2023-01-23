#include <sys/time.h>
#include "cuda_sort.h"

__global__ void gpu_mergesort(int *source, int *dest, int size, int width, int slices, dim3 *threads, dim3 *blocks);
__device__ void gpu_bottomUpMerge(int *source, int *dest, int start, int middle, int end)

    void mergesort(int *data, int size)
{

    //
    // Allocate two arrays on the GPU
    // we switch back and forth between them during the sort
    //
    int *D_data;
    int *D_swp;
    dim3 *D_threads;
    dim3 *D_blocks;

    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 8;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

    // Actually allocate the two arrays
    tm();
    (cudaMalloc((void **)&D_data, size * sizeof(int)));
    (cudaMalloc((void **)&D_swp, size * sizeof(int)));
    // if (verbose)
    //     std::cout << "cudaMalloc device lists: " << tm() << " microseconds\n";

    // Copy from our input list into the first array
    (cudaMemcpy(D_data, data, size * sizeof(int), cudaMemcpyHostToDevice));
    // if (verbose)
    //     std::cout << "cudaMemcpy list to device: " << tm() << " microseconds\n";

    //
    // Copy the thread / block info to the GPU as well
    //
    (cudaMalloc((void **)&D_threads, sizeof(dim3)));
    (cudaMalloc((void **)&D_blocks, sizeof(dim3)));

    // if (verbose)
    //     std::cout << "cudaMalloc device thread data: " << tm() << " microseconds\n";
    (cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice));
    (cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice));

    // if (verbose)
    //     std::cout << "cudaMemcpy thread data to device: " << tm() << " microseconds\n";

    int *A = D_data;
    int *B = D_swp;

    int nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                   blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    //
    // Slice up the list and give pieces of it to each thread, letting the pieces grow
    // bigger and bigger until the whole list is sorted
    //
    for (int width = 2; width < (size << 1); width <<= 1)
    {
        int slices = size / ((nThreads)*width) + 1;

        // if (verbose)
        // {
        //     std::cout << "mergeSort - width: " << width
        //               << ", slices: " << slices
        //               << ", nThreads: " << nThreads << '\n';
        //     tm();
        // }

        // Actually call the kernel
        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);

        // if (verbose)
        //     std::cout << "call mergesort kernel: " << tm() << " microseconds\n";

        // Switch the input / output arrays instead of copying them around
        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }

    //
    // Get the list back from the GPU
    //
    tm();
    (cudaMemcpy(data, A, size * sizeof(int), cudaMemcpyDeviceToHost));
    // if (verbose)
    //     std::cout << "cudaMemcpy list back to host: " << tm() << " microseconds\n";

    // Free the GPU memory
    (cudaFree(A));
    (cudaFree(B));
    // if (verbose)
    //     std::cout << "cudaFree: " << tm() << " microseconds\n";
}

// GPU helper function
// calculate the id of the current thread
__device__ unsigned int getIdx(dim3 *threads, dim3 *blocks)
{
    int x;
    return threadIdx.x +
           threadIdx.y * (x = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x * (x *= threads->z) +
           blockIdx.y * (x *= blocks->z) +
           blockIdx.z * (x *= blocks->y);
}

//
// Perform a full mergesort on our section of the data.
//
__global__ void gpu_mergesort(int *source, int *dest, int size, int width, int slices, dim3 *threads, dim3 *blocks)
{
    unsigned int idx = getIdx(threads, blocks);
    int start = width * idx * slices,
        middle,
        end;

    for (int slice = 0; slice < slices; slice++)
    {
        if (start >= size)
            break;

        middle = min(start + (width >> 1), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}

//
// Finally, sort something
// gets called by gpu_mergesort() for each slice
//
__device__ void gpu_bottomUpMerge(int *source, int *dest, int start, int middle, int end)
{
    int i = start;
    int j = middle;
    for (int k = start; k < end; k++)
    {
        if (i < middle && (j >= end || source[i] < source[j]))
        {
            dest[k] = source[i];
            i++;
        }
        else
        {
            dest[k] = source[j];
            j++;
        }
    }
}

//
// Get the time (in microseconds) since the last call to tm();
// the first value returned by this must not be trusted
//
timeval tStart;
int tm()
{
    timeval tEnd;
    gettimeofday(&tEnd, 0);
    int t = (tEnd.tv_sec - tStart.tv_sec) * 1000000 + tEnd.tv_usec - tStart.tv_usec;
    tStart = tEnd;
    return t;
}