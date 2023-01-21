#include "sort.h"
#include <string.h>
#include <stdlib.h>

clock_t start, end;
double elapsed_time;
// int numberOfComparisons;
// int numberOfSwaps;

void top_down_merge(int *array, int start, int end, int *temp)
{
    int n = end - start;
    if (n < 2)
    {
        return;
    }

    int middle = (end + start) / 2;

    top_down_merge(array, start, middle, temp);
    top_down_merge(array, middle, end, temp);

    int i = start, j = middle;
    for (int index = start; index < end; index++)
    {
        if (i < middle && (j >= end || array[i] <= array[j]))
        {
            temp[index] = array[i];
            i++;
        }
        else
        {
            temp[index] = array[j];
            j++;
        }
    }

    memcpy((array + start), (temp + start), sizeof(int) * (n));
}

void merge_sort(int *array, int number_of_elements)
{
    int *temp = malloc(number_of_elements * sizeof(int));
    top_down_merge(array, 0, number_of_elements, temp);
    free(temp);
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

int get_elapsed_time()
{
    return elapsed_time;
}