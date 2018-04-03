#include <mpi.h>
#include <math.h>
#include "calculate_sf.h"

extern "C" void calculate_sf(double** cell, double** coords, double* atom_i, double** params, int natoms, int nsyms) { //, double** symf, double** dsymf) {
    // originally, dsymf is 4D array
    // coords are fractional coordinates of atoms

    int total_bins, max_atoms_bin, bin_num, nneigh;
    int bin_range[3], nbins[3], cell_shift[3], max_bin[3], min_bin[3], pbc_bin[3], total_shift[3];
    int bin_i[natoms][4];
    double vol, tmp, cutoff, tmp_dist;
    double plane_d[3];
    double cross[3][3], reci[3][3];
    //double symf[natoms][nsyms];
    //double dsymf[natoms*natoms][nsyms*3]; // tmp setting
    //double *neigh_list;

    cutoff = params[0][0];
    total_bins = 1;

    // calculate the distance between cell plane
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
        for (int j=0; j<3; j++) {
            reci[i][j] = cross[i][j]/vol;
            tmp += reci[i][j]*reci[i][j];
        }
        plane_d[i] = 1/sqrt(tmp);
        nbins[i] = ceil(plane_d[i]/cutoff);
        total_bins *= nbins[i];
    }

    int *atoms_bin = new int[total_bins];
    for (int i=0; i<total_bins; i++)
        atoms_bin[i] = 0;

    // assign the bin index to each atom
    for (int i=0; i<natoms; i++) {
        for (int j=0; j<3; j++) {
            bin_i[i][j] = (int) coords[i][j]*nbins[j];
        }
        bin_i[i][3] = bin_i[i][0] + nbins[0]*bin_i[i][1] + nbins[0]*nbins[1]*bin_i[i][2];
        atoms_bin[bin_i[i][3]]++;
    }

    max_atoms_bin = 0;
    for (int i=0; i < total_bins; i++) {
        if (atoms_bin[i] > max_atoms_bin)
            max_atoms_bin = atoms_bin[i];
    }

    delete[] atoms_bin;

    // # of bins in each direction
    for (int i=0; i < 3; i++)
        bin_range[i] = ceil(cutoff * nbins[i] / plane_d[i]);

    // parallelize using mpi
    //

    for (int i=0; i < natoms; i++) {
        // calculate neighbor atoms

        double* nei_list = new double[max_atoms_bin * 4];
        nneigh = 0;
        
        for (int j=0; j < 3; j++) {
            max_bin[j] = bin_i[i][j] + bin_range[j];
            min_bin[j] = bin_i[i][j] - bin_range[j];
        }

        for (int dx=min_bin[0]; dx < max_bin[0]+1; dx++) {
            for (int dy=min_bin[1]; dy < max_bin[1]+1; dy++) {
                for (int dz=min_bin[2]; dz < max_bin[2]+1; dz++) {
                    for (int j=0; j < 3; j++) {
                        pbc_bin[j] = (dx%nbins[j] + nbins[j]) % nbins[j];
                        cell_shift[j] = (dx-pbc_bin[j]) / nbins[j];
                    }
                    bin_num = pbc_bin[0] + nbins[0]*pbc_bin[1] + nbins[0]*nbins[1]*pbc_bin[2];

                    for (int j=0; j < natoms; j++) {
                        if (bin_i[j][3] != bin_num)
                            continue;

                        for (int a=0; a < 3; a++) {
                            total_shift[a] = cell_shift[0]*cell[0][a] + cell_shift[1]*cell[1][a] + cell_shift[2]*cell[2][a]
                                             + coords[j][0] - coords[i][0]; // change fractional coordinates to cartesian
                        }

                        tmp_dist = sqrt(total_shift[0]*total_shift[0] + total_shift[1]*total_shift[1] + total_shift[2]*total_shift[2]);

                        if (tmp_dist < cutoff) {
                            for (int a=0; a < 3; a++) 
                                nei_list[nneigh*4 + a] = total_shift[a];
                            nneigh++;
                            
                        }
                    }
                }
            }
        }

/*
        for (int j=0; j < nneigh-1; j++) {
            // calculate radial symmetry function
            for (int k=j; k < nneigh; k++) {
                // calculate angular symmetry function
            }
        }
*/
        delete[] nei_list;
    }

}

