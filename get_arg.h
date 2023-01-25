#ifndef GET_OPT
#define GET_OPT

#include "sort.h"
#include "array.h"

struct opt
{
    int size;
    int array_type;
    enum SortMethod method;
};

typedef struct opt Options;

int check_arguments(int argc);
void get_args(int argc, char **argv, Options *options);
int get_sort_method(char method[]);
int get_array_type(char type[]);
int get_array_size(char size_of_options[]);

#endif