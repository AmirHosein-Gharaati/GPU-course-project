#ifndef SORT
#define SORT

enum SortMethod
{
    MERGE = 1,
    UNDEFINED = -1
};

int *sort_array(int *array, int size, enum SortMethod method);

#endif