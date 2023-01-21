#include "sort.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

double elapsed_time;

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
    struct timeval start, end;

    switch (method)
    {
    case MERGE:
        gettimeofday(&start, NULL);
        merge_sort(array, size);
        gettimeofday(&end, NULL);
        break;

    default:
        break;
    }

    elapsed_time = (end.tv_sec - start.tv_sec) * 1000.0;
    elapsed_time += (end.tv_usec - start.tv_usec) / 1000.0;
    return array;
}

double get_elapsed_time()
{
    return elapsed_time;
}