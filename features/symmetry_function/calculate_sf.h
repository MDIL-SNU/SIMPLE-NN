/*
 Code for calculating symmetry function value
 For that, this code recieve the atomic coordinates and calculate nearest neighbor.
 In LAMMPS code, this part is included in pair_nn.cpp
 */

#include <math.h>
//#include "mpi.h"
#include "symmetry_functions.h"

extern "C" void calculate_sf(double **, double **, double **, int *, double **, int, int); //, double**, double**);
