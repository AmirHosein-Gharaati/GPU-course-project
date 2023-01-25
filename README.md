# Parallel Sorting with CUDA

Objective: The goal of this project is to use CUDA to implement a parallel sorting algorithm and compare its performance to a sequential version.

Motivation: Sorting large datasets can be computationally intensive, and using CUDA to parallelize the computation can improve performance and allow for larger datasets to be sorted.

Methodology:

- Choose a sorting algorithm to implement using CUDA (e.g. merge sort, radix sort).
- Write a CUDA kernel to perform the sorting algorithm in parallel. The kernel should take as input the data to be sorted and the sorting criteria (e.g. ascending or descending order). It should output the sorted data.
- Write a host function to launch the kernel and measure the performance of the CUDA implementation.
- Write a sequential version of the sorting algorithm and measure its performance.
- Compare the performance of the CUDA and sequential implementations and analyze the speedup.
- Analyze the scalability of the CUDA implementation by sorting datasets of increasing size and comparing the performance.

Expected Results:

- The CUDA implementation should be significantly faster than the sequential version, especially for large datasets.
- The CUDA implementation should scale well with increasing dataset size.

Conclusion: By using CUDA to parallelize the calculation of the sorting algorithm, we were able to significantly improve the performance of the algorithm. This allowed us to sort larger datasets in a shorter amount of time and achieve good scalability.

Repository of sorting algorithms in C and CUDA.

## Information

> Our program generates and fills arrays in four different ways:

1. arrays with totally random elements
2. arrays already ordered
3. arrays ordered in descending order
4. arrays 90% ordered.

> Sorting methods implemented

1. Merge sort
2. CUDA Merge sort

## Requirements

> NVIDIA CUDA Toolkit 6.0, NVCC v6.0.1, GCC and G++

Follow these instructions to set up your environment:
[prosciens’s tutorial to set up CUDA 6 compiler environment on Debian testing/sid](http://prosciens.com/prosciens/how-to-install-nvidia-cuda-6-and-compile-all-the-samples-in-debian-testing-x86_64/ "prosciens’s instructions")

CUDA sorting code requires devices with CUDA compute capability 3.5 or higher, in order to use
the Dinamic Parallelism technology, read more about it here:

[NVIDIA blog describing Dinamic Parallelism in Kepler GPUs](http://blogs.nvidia.com/blog/2012/09/12/how-tesla-k20-speeds-up-quicksort-a-familiar-comp-sci-code/ "NVIDIA blog")

## Compiling

Run the MAKEFILE

## Instructions

To run the program, type:

```c
./a.out -a $algorithm -n $number_of_elements -s $state
```

### Parameters

> 1. -a sorting algorithm
> 2. -n number of elements
> 3. -s array state

| Param |   Value    |
| ----- | :--------: |
| -a    |   merge    |
|       |  gpumerge  |
| -n    |  int > 0   |
| -s    |   random   |
|       | ascending  |
|       | descending |
|       |   almost   |
