#ifndef ARRAY_OPERATIONS
#define ARRAY_OPERATIONS 1

#define RANDOM_ORDER 0
#define ASCENDING_ORDER 1
#define DESCENDING_ORDER 2
#define ALMOST_ORDERED 3

int *generate_array(int size, int array_type);
void init_random_array(int *array, int size);

#endif