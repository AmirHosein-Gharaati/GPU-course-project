#include "sort.h"
#include <string.h>
#include <stdlib.h>

clock_t start, end;
double elapsed_time;
// int numberOfComparisons;
// int numberOfSwaps;

void merge_sort(int *array, int number_of_elements)
{
}

/**
 * Method that receives a pointer to an array that will be sorted,
 * his size and the enum of the method that will be used
 * @param  array  Array to be sorted
 * @param  size   Size of the array
 * @param  method Sorting algorithm enum
 * @return        Pointer to the sorted array
 */
int *sort_array(int *array, int size, enum SortMethod method)
{
    switch (method)
    {
    case MERGE:
        start = clock();
        merge_sort(array, size);
        end = clock();
        break;

    default:
        break;
    }

    elapsed_time = (((double)(end - start)) / CLOCKS_PER_SEC);
    return array;
}