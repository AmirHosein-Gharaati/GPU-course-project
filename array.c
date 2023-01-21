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

        case ASCENDING_ORDER:
            init_ascending_array(arr, size);
            break;

        case DESCENDING_ORDER:
            init_descending_array(arr, size);

        case ALMOST_ORDERED:
            init_almost_ordered_array(arr, size);

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

/**
 * Function that fills an array with integers in ascending order
 * @param int* array Reference to the array that will be filled
 * @param int  size  Number of elements
 */
void init_ascending_array(int *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        array[i] = i;
    }
}

/**
 * Function that fills an array with integers in ascending order
 * @param int* array Reference to the array that will be filled
 * @param int  size  Number of elements
 */
void init_descending_array(int *array, int size)
{
    int i, j;
    for (i = 0, j = size; i < size; i++, j--)
    {
        array[i] = j;
    }
}

/**
 * Function that fills 90% of an array with integers in ascending order
 * an the other 10% with random integers
 * @param int* array Reference to the array that will be filled
 * @param int  size  Number of elements
 */
void init_almost_ordered_array(int *array, int size)
{
    srand(time(NULL));
    int ninety_percent_size = (int)size / 10 * 9;
    int i = 0;
    for (i = 1; i <= ninety_percent_size; i++)
    {
        array[i] = i;
    }

    for (i = ninety_percent_size; i < size; i++)
    {
        array[i] = rand() % (size - ninety_percent_size) + ninety_percent_size;
    }
}