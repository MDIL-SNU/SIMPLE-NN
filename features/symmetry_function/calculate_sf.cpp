#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include "calculate_sf.h"

extern "C" void calculate_sf(double** cell, double** cart, double** scale, int* atom_i, 
                             double** params, int natoms, int nsyms,
                             double** symf, double** dsymf) {
    // TODO: add the part for calculating symmetry function
    // TODO: add the part for parallelization
    // originally, dsymf is 4D array
    // scale are fractional coordinates of atoms

    int total_bins, max_atoms_bin, bin_num, neigh_check_bins, nneigh;
    int bin_range[3], nbins[3], cell_shift[3], max_bin[3], min_bin[3], pbc_bin[3];
    int bin_i[natoms][4];
    double vol, tmp, cutoff, cutoff_sq, tmp_sq;
    double plane_d[3], total_shift[3], precal[11], vecij[3], vecik[3], vecjk[3], tmpd[9];
    double cross[3][3], reci[3][3];
    //double symf[natoms][nsyms];
    //double dsymf[natoms*natoms][nsyms*3]; // tmp setting
    
    cutoff = params[0][0]; // FIXME: Find the maximum cutoff of parameter list
    cutoff = 0.0;
    for (int s=0; s < nsyms; ++s) {
        if (cutoff < params[s][])
            cutoff = params[s][];
    }

    cutoff_sq = cutoff*cutoff;
    printf("cutoff: %f\n", cutoff);
    printf("Cell info:\n  %f %f %f\n  %f %f %f\n  %f %f %f\n", cell[0][0], cell[0][1], cell[0][2],
                                                               cell[1][0], cell[1][1], cell[1][2],
                                                               cell[2][0], cell[2][1], cell[2][2]);
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
    for (int i=0; i<3; ++i) {
        tmp = 0;
        for (int j=0; j<3; ++j) {
            reci[i][j] = cross[i][j]/vol;
            tmp += reci[i][j]*reci[i][j];
        }
        plane_d[i] = 1/sqrt(tmp);
        nbins[i] = ceil(plane_d[i]/cutoff);
        total_bins *= nbins[i];
    }
    printf("plane distance: %f %f %f\n", plane_d[0], plane_d[1], plane_d[2]);
    printf("# of bins: %d %d %d\n", nbins[0], nbins[1], nbins[2]);
    printf("total bins: %d\n", total_bins);

    int *atoms_bin = new int[total_bins];
    for (int i=0; i<total_bins; ++i)
        atoms_bin[i] = 0;

    // assign the bin index to each atom
    for (int i=0; i<natoms; ++i) {
        for (int j=0; j<3; ++j) {
            bin_i[i][j] = scale[i][j] * (double) nbins[j];
        }
        bin_i[i][3] = bin_i[i][0] + nbins[0]*bin_i[i][1] + nbins[0]*nbins[1]*bin_i[i][2];
        atoms_bin[bin_i[i][3]]++;
    }

    max_atoms_bin = 0;
    for (int i=0; i < total_bins; ++i) {
        if (atoms_bin[i] > max_atoms_bin)
            max_atoms_bin = atoms_bin[i];
    }

    printf("max_atoms_bin: %d\n", max_atoms_bin);

    delete[] atoms_bin;

    // # of bins in each direction
    neigh_check_bins = 1;
    for (int i=0; i < 3; ++i) {
        bin_range[i] = ceil(cutoff * nbins[i] / plane_d[i]);
        neigh_check_bins *= 2*bin_range[i];
    }
    printf("%d %d %d\n", bin_range[0], bin_range[1], bin_range[2]);

    // parallelize using mpi
    //

    for (int i=0; i < natoms; ++i) {
        // calculate neighbor atoms
        printf("atom %d\n", i+1);
        double* nei_list = new double[max_atoms_bin * 5 * neigh_check_bins];
        nneigh = 0;
        
        for (int j=0; j < 3; ++j) {
            max_bin[j] = bin_i[i][j] + bin_range[j];
            min_bin[j] = bin_i[i][j] - bin_range[j];
        }

        for (int dx=min_bin[0]; dx < max_bin[0]+1; ++dx) {
            for (int dy=min_bin[1]; dy < max_bin[1]+1; ++dy) {
                for (int dz=min_bin[2]; dz < max_bin[2]+1; ++dz) {
                    pbc_bin[0] = (dx%nbins[0] + nbins[0]) % nbins[0];
                    pbc_bin[1] = (dy%nbins[1] + nbins[1]) % nbins[1];
                    pbc_bin[2] = (dz%nbins[2] + nbins[2]) % nbins[2];
                    cell_shift[0] = (dx-pbc_bin[0]) / nbins[0];
                    cell_shift[1] = (dy-pbc_bin[1]) / nbins[1];
                    cell_shift[2] = (dz-pbc_bin[2]) / nbins[2];
                    
                    bin_num = pbc_bin[0] + nbins[0]*pbc_bin[1] + nbins[0]*nbins[1]*pbc_bin[2];
                    
                    for (int j=0; j < natoms; ++j) {
                        if (bin_i[j][3] != bin_num)
                            continue;

                        if (!(cell_shift[0] || cell_shift[1] || cell_shift[2]) && (i == j))
                            continue;

                        for (int a=0; a < 3; ++a) {
                            total_shift[a] = cell_shift[0]*cell[0][a] + cell_shift[1]*cell[1][a] + cell_shift[2]*cell[2][a]
                                             + cart[j][a] - cart[i][a];
                        }

                        tmp_sq = total_shift[0]*total_shift[0] + total_shift[1]*total_shift[1] + total_shift[2]*total_shift[2];
                        
                        if (tmp_sq < cutoff_sq) {
                            printf("  %f %f %f | %f %f %f | %d %d %d | %f %f %f | %f\n", cart[i][0], cart[i][1], cart[i][2], 
                                    cart[j][0], cart[j][1], cart[j][2], cell_shift[0], cell_shift[1], cell_shift[2], 
                                    total_shift[0], total_shift[1], total_shift[2], tmp_sq);
                            for (int a=0; a < 3; ++a) 
                                nei_list[nneigh*5 + a] = total_shift[a];
                            nei_list[nneigh*5 + 3] = tmp_sq;
                            nei_list[nneigh*5 + 4] = atom_i[j];
                            nneigh++;
                        }
                    }
                }
            }
        }

        for (int j=0; j < nneigh-1; ++j) {
            // calculate radial symmetry function

            rRij = sqrt(neigh_list[j*5 + 3]);
            vecij[0] =  neigh_list[j*5]     / rRij;
            vecij[1] =  neigh_list[j*5 + 1] / rRij;
            vecij[2] =  neigh_list[j*5 + 2] / rRij;

            for (int s=0; s < nsyms; ++s) {
                if ([radial_symfunc] && [target atom type]) { // FIXME:
                    precal[0] = cutf();
                    precal[1] = dcutf();

                    symf[][] += G2(); // FIXME: 
                    tmpd[0] = ;
                    tmpd[1] = ;
                    tmpd[2] = ;
                }
                else continue;
            }

            for (int k=j; k < nneigh; ++k) {
                // calculate angular symmetry function
                rRik = sqrt(neigh_list[k*5 + 3]);
                vecik[0] = neigh_list[k*5]     / rRik;
                vecik[1] = neigh_list[k*5 + 1] / rRik;
                vecik[2] = neigh_list[k*5 + 2] / rRik;

                deljk[0] = ;
                deljk[1] = ;
                deljk[2] = ;
                rRik = sqrt(neigh_list[k*5 + 3]);
                vecik[0] = neigh_list[k*5]     / rRik;
                vecik[1] = neigh_list[k*5 + 1] / rRik;
                vecik[2] = neigh_list[k*5 + 2] / rRik;

                for (int s=0; s < nsyms; ++s) {
                    if ([angular_symfunc] && [target atom type]) { // FIXME:
                        symf[][] += G4();

                        tmpd[0] = ;
                        tmpd[1] = ;
                        tmpd[2] = ;
                        tmpd[3] = ;
                        tmpd[4] = ;
                        tmpd[5] = ;
                        tmpd[6] = ;
                        tmpd[7] = ;
                        tmpd[8] = ;
                    }
                }
            }
        }

        delete[] nei_list;
    }

}
