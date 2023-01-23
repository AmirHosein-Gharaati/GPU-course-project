#include <cuda_sort.h>
#include <sys/time.h>

void mergesort(long *data, long size, dim3 threadsPerBlock, dim3 blocksPerGrid)
{

    //
    // Allocate two arrays on the GPU
    // we switch back and forth between them during the sort
    //
    long *D_data;
    long *D_swp;
    dim3 *D_threads;
    dim3 *D_blocks;

    // Actually allocate the two arrays
    tm();
    checkCudaErrors(cudaMalloc((void **)&D_data, size * sizeof(long)));
    checkCudaErrors(cudaMalloc((void **)&D_swp, size * sizeof(long)));
    if (verbose)
        std::cout << "cudaMalloc device lists: " << tm() << " microseconds\n";

    // Copy from our input list into the first array
    checkCudaErrors(cudaMemcpy(D_data, data, size * sizeof(long), cudaMemcpyHostToDevice));
    if (verbose)
        std::cout << "cudaMemcpy list to device: " << tm() << " microseconds\n";

    //
    // Copy the thread / block info to the GPU as well
    //
    checkCudaErrors(cudaMalloc((void **)&D_threads, sizeof(dim3)));
    checkCudaErrors(cudaMalloc((void **)&D_blocks, sizeof(dim3)));

    if (verbose)
        std::cout << "cudaMalloc device thread data: " << tm() << " microseconds\n";
    checkCudaErrors(cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice));

    if (verbose)
        std::cout << "cudaMemcpy thread data to device: " << tm() << " microseconds\n";

    long *A = D_data;
    long *B = D_swp;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    //
    // Slice up the list and give pieces of it to each thread, letting the pieces grow
    // bigger and bigger until the whole list is sorted
    //
    for (int width = 2; width < (size << 1); width <<= 1)
    {
        long slices = size / ((nThreads)*width) + 1;

        if (verbose)
        {
            std::cout << "mergeSort - width: " << width
                      << ", slices: " << slices
                      << ", nThreads: " << nThreads << '\n';
            tm();
        }

        // Actually call the kernel
        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, size, width, slices, D_threads, D_blocks);

        if (verbose)
            std::cout << "call mergesort kernel: " << tm() << " microseconds\n";

        // Switch the input / output arrays instead of copying them around
        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }

    //
    // Get the list back from the GPU
    //
    tm();
    checkCudaErrors(cudaMemcpy(data, A, size * sizeof(long), cudaMemcpyDeviceToHost));
    if (verbose)
        std::cout << "cudaMemcpy list back to host: " << tm() << " microseconds\n";

    // Free the GPU memory
    checkCudaErrors(cudaFree(A));
    checkCudaErrors(cudaFree(B));
    if (verbose)
        std::cout << "cudaFree: " << tm() << " microseconds\n";
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
__global__ void gpu_mergesort(long *source, long *dest, long size, long width, long slices, dim3 *threads, dim3 *blocks)
{
    unsigned int idx = getIdx(threads, blocks);
    long start = width * idx * slices,
         middle,
         end;

    for (long slice = 0; slice < slices; slice++)
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
__device__ void gpu_bottomUpMerge(long *source, long *dest, long start, long middle, long end)
{
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++)
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

// read data into a minimal linked list
typedef struct
{
    int v;
    void *next;
} LinkNode;

// helper function for reading numbers from stdin
// it's 'optimized' not to check validity of the characters it reads in..
long readList(long **list)
{
    tm();
    long v, size = 0;
    LinkNode *node = 0;
    LinkNode *first = 0;
    while (std::cin >> v)
    {
        LinkNode *next = new LinkNode();
        next->v = v;
        if (node)
            node->next = next;
        else
            first = next;
        node = next;
        size++;
    }

    if (size)
    {
        *list = new long[size];
        LinkNode *node = first;
        long i = 0;
        while (node)
        {
            (*list)[i++] = node->v;
            node = (LinkNode *)node->next;
        }
    }

    if (verbose)
        std::cout << "read stdin: " << tm() << " microseconds\n";

    return size;
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