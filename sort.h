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
void top_down_merge(int *array, int start, int end, int *temp);
void merge_sort(int *array, int number_of_elements);
int get_elapsed_time();
#endif