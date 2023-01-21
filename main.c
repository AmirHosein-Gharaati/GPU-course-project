#include <stdio.h>
#include <stdlib.h>
#include "get_arg.h"

int main(int argc, char **argv)
{

    // if (!check_arguments(argc))
    // {
    //     fprintf(stderr, "Not enough parameters\n");
    //     fprintf(stderr, "Expected: -a SORTTYPE -n ARRAYSIZE -s ARRAYSTATE [-P]\n");
    //     fprintf(stderr, "Please read the doc.\n");
    //     return 1;
    // }

    Options *options = (struct opt *)malloc(sizeof(Options));

    get_args(argc, argv, options);

    int *array = generate_array(options->size, options->array_type);

    int *array_copy = clone_array(array, options->size);

    // sort array

    // print arrays

    free(options);
    free(array);
    free(array_copy);

    return 0;
}