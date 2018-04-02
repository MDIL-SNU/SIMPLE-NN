#include <mpi.h>
#include <math.h>
#include "symmetry_functions.h"
#include "calculate_sf.h"

extern "C" void calculate_sf(double** cell, double** coords, double* atom_i, double** params, int natoms, int nsyms, double** symf, double** dsymf) {
    // originally, dsymf is 4D array
    // coords are fractional coordinates of atoms

    int n, n_bin, search_bins_x, search_bins_y, search_bins_z;
    double a, vol, tmp, cutoff;
    double reci[3][3];
    double cross[3][3];
    double plane_d[3];
    int nbins[3];
    int bin_i[natoms][3];
    double symf[natoms][nsyms];
    double dsymf[natoms*natoms][nsyms*3]; // add dimension
    double *neigh_list;

    cutoff = params[0][0];
    n_bin = 1;

    cross[0][0] = cell[1][1]*cell[2][2] - cell[1][2]*cell[2][1];
    cross[0][1] = cell[1][2]*cell[2][0] - cell[1][0]*cell[2][2];
    cross[0][2] = cell[1][0]*cell[2][1] - cell[1][1]*cell[2][0];
    cross[1][0] = cell[2][1]*cell[0][2] - cell[2][2]*cell[0][1];
    cross[1][1] = cell[2][2]*cell[0][0] - cell[2][0]*cell[0][2];
    cross[1][2] = cell[2][0]*cell[0][1] - cell[2][1]*cell[0][0];
    cross[2][0] = cell[0][1]*cell[1][2] - cell[0][2]*cell[1][1];
    cross[2][1] = cell[0][2]*cell[1][0] - cell[0][0]*cell[1][2];
    cross[2][2] = cell[0][0]*cell[1][1] - cell[0][1]*cell[1][0];

    vol = cross[0][0]*cell[0][0] + cross[0][1]*cell[0][1] + cross[0][2]*cell[0][2];
    for (int i=0; i<3; i++) {
        tmp = 0;
        for (int j=0; j<3; j++)
            reci[i][j] = cross[i][j]/vol;
            tmp += reci[i][j]*reci[i][j];
        plane_d[i] = 1/sqrt(tmp);
        nbins[i] = ceil(plane_d[i]/cutoff);
        n_bin *= nbins[i];
    }

    // assign the bin index to each atom
    for (int i=0; i<natoms; i++) {
        for (int j=0; j<3, j++) {
            bin_i[i][j] = (int) coords[i][j]*nbins[j];
        }
    }

    // # of bins in each direction
    search_bins_x = ceil(cutoff * nbins / plane_d[0]);
    search_bins_y = ceil(cutoff * nbins / plane_d[1]);
    search_bins_z = ceil(cutoff * nbins / plane_d[2]);
    
    // parallelize using mpi

    for (int i=0; i < natoms; i++) {
        // calculate neighbor atoms
        for (int dx=-search_bins_x; dx < search_bins_x+1; dx++) {
            for (int dy=-search_bins_y; dy < search_bins_y+1; dy++) {
                for (int dz=-search_bins_z; dz < search_bins_z+1; dz++) {
                    for (int j=0; j < natoms; j++) {
                        cell_shift[0] = floor((bin_i[j][0] + dx)/nbins[0]);
                        cell_shift[1] = floor((bin_i[j][1] + dy)/nbins[1]);
                        cell_shift[2] = floor((bin_i[j][2] + dz)/nbins[2]);

                        if (bin_i[j][0]] + cell_shift[0] - )
                            continue;

                        total_shift[0] = cell_shift[0]*cell[0][0] + cell_shift[1]*cell[1][0] + cell_shift[2]*cell[2][0]
                                         + coords[j][0] - coords[i][0];
                        total_shift[1] = cell_shift[0]*cell[0][1] + cell_shift[1]*cell[1][1] + cell_shift[2]*cell[2][1]
                                         + coords[j][1] - coords[i][1];
                        total_shift[2] = cell_shift[0]*cell[0][2] + cell_shift[1]*cell[1][2] + cell_shift[2]*cell[2][2]
                                         + coords[j][2] - coords[j][2];
                        tmp_dist = sqrt(total_shift[0]*total_shift[0] + total_shift[1]*total_shift[1] + total_shift[2]*total_shift[2]);

                        if (tmp_dist < cutoff) {
                            neigh_list[nneigh][0] = total_shift[0];
                            neigh_list[nneigh][1] = total_shift[1];
                            neigh_list[nneigh][2] = total_shift[2];
                            nneigh++;
                        }
                    }
                }
            }
        }


        for (int j=0; j < nneigh-1; j++) {
            // calculate radial symmetry function
            for (int k=j; k < nneigh; k++) {
                // calculate angular symmetry function
            }
        }
    }

}

//double* get_neighborlist() {
//}

