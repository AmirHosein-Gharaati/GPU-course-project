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

    return 0;
}