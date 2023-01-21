#include "get_arg.h"
#include <getopt.h>

static const char *optString = "a:n:s:P";

int check_arguments(int argc)
{
    return argc > 6;
}

/**
 * Function that receives the args and calls the program with the
 * given parameters
 * @param int       argc
 * @param char**    argv
 * @param Options*  options
 */
void get_args(int argc, char **argv, Options *options)
{
    int opt = 0;

    while ((opt = getopt(argc, argv, optString)) != -1)
    {
        switch (opt)
        {
        case 'a': // algorithm
            options->method = get_sort_method(optarg);
            break;
        case 'n': // number of elements
            options->size = get_array_size(optarg);
            break;

        case 's': // situation
            options->array_type = get_array_type(optarg);

        case 'P': // print
            options->print_vector = 1;
            break;

        default:
            break;
        }
    }
}

/**
 * Function that receives a string which is a sorting method name
 * and returns his constant value
 * @param  char[] method
 * @return int
 */
int get_sort_method(char method[])
{
    int selected_method;

    if (strcmp(method, "merge") == 0)
        selected_method = MERGE;
    else
        selected_method = UNDEFINED;

    return selected_method;
}

/**
 * Function that receives a string containing
 * a number and convert it to an integer
 * @param  char[] method
 * @return int
 */
int get_array_size(char size_of_options[])
{
    int size = atoi(size_of_options);
    if (size < 0)
        size = UNDEFINED;
    return size;
}

/**
 * Function that receives a string which is an array type
 * and returns his constant
 * @param  char[] type
 * @return int
 */
int get_array_type(char type[])
{
    int selected_type;
    if (strcmp(type, "random") == 0)
        selected_type = RANDOM_ORDER;
    else if (strcmp(type, "ascending") == 0)
        selected_type = ASCENDING_ORDER;
    else if (strcmp(type, "descending") == 0)
        selected_type = DESCENDING_ORDER;
    else if (strcmp(type, "almost") == 0)
        selected_type = ALMOST_ORDERED;
    else
        selected_type = UNDEFINED;
    return selected_type;
}