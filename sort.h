#ifndef SORT
#define SORT

#include <time.h>

enum SortMethod
{
    MERGE = 1,
    UNDEFINED = -1
};

// extern int numberOfComparisons;
// extern int numberOfSwaps;
extern clock_t start, end;
extern double elapsed_time;

int *sort_array(int *array, int size, enum SortMethod method);

#endif