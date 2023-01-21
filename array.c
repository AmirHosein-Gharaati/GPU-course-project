#include "array.h"
#include <stdlib.h>
#include <time.h>
#include <math.h>

/**
 * Function that receives the size and the array_type of the
 * required array and returns it filled.
 * @param  int   size          Size of the array
 * @param  int   array_type  How the array will be filled
 * @return int[]               Filled array
 */
int *generate_array(int size, int array_type)
{
    int *arr = malloc(size * sizeof(int));

    if (arr)
    {
        switch (array_type)
        {
        case RANDOM_ORDER:
            init_random_array(arr, size);
            break;

        default:
            break;
        }
    }

    return arr;
}

/**
 * Function that fills an array with random integers
 * @param int* array Reference to the array that will be filled
 * @param int  size  Number of elements
 */
void init_random_array(int *array, int size)
{
    srand(time(NULL));
    int i;
    for (i = 0; i < size; i++)
    {
        array[i] = rand() % size;
    }
}