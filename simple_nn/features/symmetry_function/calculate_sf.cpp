//#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include "calculate_sf.h"

extern "C" int calculate_sf(double** cell, double** cart, double** scale,
                            int* atom_i, int natoms, int* cal_atoms, int cal_num,
                            int** params_i, double** params_d, int nsyms,
                            double** symf, double** dsymf, double** dsymfa) {
    // cell: cell info of structure
    // cart: cartesian coordinates of atoms
    // scale: fractional coordinates of atoms
    // atom_i: atom type index (start with 1)
    // params_i: integer parameter for symmetry function
    //           [symmetry function type, 1st neighbor atom type, 2nd neighbor atom type]
    // params_d: double parameter for symmetry function
    //           [cutoff, param1, param2, param3]
    // natoms: # of atoms
    // nsyms: # of symmetry functions
    // symf: symmetry function vector ([# of atoms, # of symfuncs])
    // dsymf: derivative of symmetry function vector 
    // originally, dsymf is 4D array (dimension: [# of atoms, # of symfuncs, # of atoms, 3])
    // in this function, we use 2D array ([# of atoms, # of symfuncs * # of atoms * 3])
    // Similarly, dsymfa is 4D array (dimension: [# of atoms, # of symfuncs, 3, 6])
    // in this function, we use 2D array ([# of atoms, # of symfuncs * 3 * 6])

    int total_bins, max_atoms_bin, bin_num, neigh_check_bins, nneigh;
    int bin_range[3], nbins[3], cell_shift[3], max_bin[3], min_bin[3], pbc_bin[3];
    //int bin_i[natoms][4];
    double vol, tmp, cutoff, dradtmp, rRij, rRik, rRjk;
    double plane_d[3], total_shift[3], precal[12], tmpd[9], dangtmp[3];
    double vecij[3], vecik[3], vecjk[3], deljk[3];
    double cross[3][3], reci[3][3], inv[3][3];//, powtwo[nsyms];

    // Check for not implemented symfunc type.
    for (int s=0; s < nsyms; ++s) {
        bool implemented = false;
        for (int i=0; i < sizeof(IMPLEMENTED_TYPE) / sizeof(IMPLEMENTED_TYPE[0]); i++) {
            if (params_i[s][0] == IMPLEMENTED_TYPE[i]) {
                implemented = true;
                break;
            }
        }
        if (!implemented) return 1;
    }

    int **bin_i = new int*[natoms];
    for (int i=0; i<natoms; i++) {
        bin_i[i] = new int[4];
    }

    double *powtwo = new double[nsyms];

    cutoff = 0.0;
    for (int s=0; s < nsyms; ++s) {
        if (cutoff < params_d[s][0])
            cutoff = params_d[s][0];
        
        if ((params_i[s][0] == 4 || params_i[s][0] == 5) &&
             params_d[s][2] < 1.0)
            return 2;
        
        powtwo[s] = pow(2, 1.-params_d[s][2]);
    }
    
    total_bins = 1;

    // calculate the inverse matrix of cell and the distance between cell plane
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
        for (int j=0; j<3; ++j) {
            inv[i][j] = cross[j][i]/vol;
        }
    }
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

    delete[] atoms_bin;

    // # of bins in each direction
    neigh_check_bins = 1;
    for (int i=0; i < 3; ++i) {
        bin_range[i] = ceil(cutoff * nbins[i] / plane_d[i]);
        neigh_check_bins *= 2*bin_range[i];
    }

    //for (int i=0; i < natoms; ++i) {
    for (int ii=0; ii < cal_num; ++ii) {
        int i=cal_atoms[ii];
        // calculate neighbor atoms
        double* nei_list_d = new double[max_atoms_bin * 4 * neigh_check_bins];
        int*    nei_list_i = new int[max_atoms_bin * 2 * neigh_check_bins];
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

                        tmp = sqrt(total_shift[0]*total_shift[0] + total_shift[1]*total_shift[1] + total_shift[2]*total_shift[2]);
                        
                        if (tmp < cutoff) {
                            for (int a=0; a < 3; ++a) 
                                nei_list_d[nneigh*4 + a] = total_shift[a];
                            nei_list_d[nneigh*4 + 3] = tmp;
                            nei_list_i[nneigh*2]    = atom_i[j];
                            nei_list_i[nneigh*2 + 1] = j;
                            nneigh++;
                        }
                    }
                }
            }
        }

        for (int j=0; j < nneigh; ++j) {
            // calculate radial symmetry function
            double* lcoeff = new double[9]();
            // coeffcient of each directional vector (order : ij, ik)
            rRij = nei_list_d[j*4 + 3];

            for (int a=0; a < 3; ++a) {
                vecij[a] =  nei_list_d[j*4 + a] / rRij;
                for (int b=0; b < 3; ++b) {
                    lcoeff[a] += nei_list_d[j*4 + b] * inv[b][a];
                }
            }

            for (int s=0; s < nsyms; ++s) {
                if ((params_i[s][0] == 2) && (params_i[s][1] == nei_list_i[j*2])) { // FIXME:
                    precal[0] = cutf(rRij / params_d[s][0]);
                    precal[1] = dcutf(rRij, params_d[s][0]);

                    symf[ii][s] += G2(rRij, precal, params_d[s], dradtmp); // FIXME: index

                    for (int a=0; a < 3; ++a) {
                        tmpd[a] = dradtmp*vecij[a];
                        dsymf[ii][s*natoms*3 + nei_list_i[j*2 + 1]*3 + a] += tmpd[a];
                        dsymf[ii][s*natoms*3 + i*3 + a] -= tmpd[a];

                        for (int b=0; b < 3; ++b) {
                            dsymfa[ii][s*3*6 + b*6 + a]     += tmpd[a]*lcoeff[b]*cell[b][a]/vol;
                            dsymfa[ii][s*3*6 + b*6 + a + 3] += tmpd[a]*lcoeff[b]*cell[b][(a + 1)%3]/vol;
                        }
                    }
                }
                else continue;
            }

            for (int k=j+1; k < nneigh; ++k) {
                // calculate angular symmetry function
                deljk[0] = nei_list_d[k*4]     - nei_list_d[j*4];
                deljk[1] = nei_list_d[k*4 + 1] - nei_list_d[j*4 + 1];
                deljk[2] = nei_list_d[k*4 + 2] - nei_list_d[j*4 + 2];
                rRjk = sqrt(deljk[0]*deljk[0] + deljk[1]*deljk[1] + deljk[2]*deljk[2]);

                if (rRjk < 0.0001) continue;

                rRik = nei_list_d[k*4 + 3];
                for (int a=0; a < 3; ++a) {
                    vecik[a] = nei_list_d[k*4 + a] / rRik;
                    vecjk[a] = deljk[a] / rRjk;
                    lcoeff[a + 3] = 0;
                    lcoeff[a + 6] = 0;
                    for (int b=0; b < 3; ++b) {
                        lcoeff[a + 3] += nei_list_d[k*4 + b] * inv[b][a];
                        lcoeff[a + 6] += deljk[b] * inv[b][a];
                    }
                }

                precal[6]  = rRij*rRij+rRik*rRik+rRjk*rRjk;
                precal[7]  = (rRij*rRij + rRik*rRik - rRjk*rRjk)/2/rRij/rRik;
                precal[8]  = 0.5*(1/rRik + 1/rRij/rRij*(rRjk*rRjk/rRik - rRik));
                precal[9]  = 0.5*(1/rRij + 1/rRik/rRik*(rRjk*rRjk/rRij - rRij));
                precal[10] = -rRjk/rRij/rRik;
                precal[11] = rRij*rRij+rRik*rRik;

                for (int s=0; s < nsyms; ++s) {
                    if ((params_i[s][0] == 4) && 
                       (((params_i[s][1] == nei_list_i[j*2]) && (params_i[s][2] == nei_list_i[k*2])) || 
                        ((params_i[s][1] == nei_list_i[k*2]) && (params_i[s][2] == nei_list_i[j*2]))) ) { // FIXME:

                        precal[0] = cutf(rRij / params_d[s][0]);
                        precal[1] = dcutf(rRij, params_d[s][0]);
                        precal[2] = cutf(rRik / params_d[s][0]);
                        precal[3] = dcutf(rRik, params_d[s][0]);
                        precal[4] = cutf(rRjk / params_d[s][0]);
                        precal[5] = dcutf(rRjk, params_d[s][0]);

                        symf[ii][s] += G4(rRij, rRik, rRjk, powtwo[s], precal, params_d[s], dangtmp);

                        tmpd[0] = dangtmp[0]*vecij[0];
                        tmpd[1] = dangtmp[0]*vecij[1];
                        tmpd[2] = dangtmp[0]*vecij[2];
                        tmpd[3] = dangtmp[1]*vecik[0];
                        tmpd[4] = dangtmp[1]*vecik[1];
                        tmpd[5] = dangtmp[1]*vecik[2];
                        tmpd[6] = dangtmp[2]*vecjk[0];
                        tmpd[7] = dangtmp[2]*vecjk[1];
                        tmpd[8] = dangtmp[2]*vecjk[2];

                        for (int a=0; a < 3; ++a) {
                            dsymf[ii][s*natoms*3 + nei_list_i[j*2 + 1]*3 + a] += tmpd[a] - tmpd[a + 6];
                            dsymf[ii][s*natoms*3 + nei_list_i[k*2 + 1]*3 + a] += tmpd[a + 3] + tmpd[a + 6];
                            dsymf[ii][s*natoms*3 + i*3 + a] -= tmpd[a] + tmpd[a + 3];

                            for (int b=0; b < 3; ++b) {
                                dsymfa[ii][s*3*6 + b*6 + a]     += (tmpd[a]*lcoeff[b] + tmpd[a + 3]*lcoeff[b + 3] + tmpd[a + 6]*lcoeff[b + 6])*cell[b][a]/vol;
                                dsymfa[ii][s*3*6 + b*6 + a + 3] += (tmpd[a]*lcoeff[b] + tmpd[a + 3]*lcoeff[b + 3] + tmpd[a + 6]*lcoeff[b + 6])*cell[b][(a + 1)%3]/vol;

                            }
                        }
                    }
                    else if ((params_i[s][0] == 5) &&
                           (((params_i[s][1] == nei_list_i[j*2]) && (params_i[s][2] == nei_list_i[k*2])) ||
                            ((params_i[s][1] == nei_list_i[k*2]) && (params_i[s][2] == nei_list_i[j*2]))) ) {

                        precal[0] = cutf(rRij / params_d[s][0]);
                        precal[1] = dcutf(rRij, params_d[s][0]);
                        precal[2] = cutf(rRik / params_d[s][0]);
                        precal[3] = dcutf(rRik, params_d[s][0]);

                        symf[ii][s] += G5(rRij, rRik, powtwo[s], precal, params_d[s], dangtmp);

                        tmpd[0] = dangtmp[0]*vecij[0];
                        tmpd[1] = dangtmp[0]*vecij[1];
                        tmpd[2] = dangtmp[0]*vecij[2];
                        tmpd[3] = dangtmp[1]*vecik[0];
                        tmpd[4] = dangtmp[1]*vecik[1];
                        tmpd[5] = dangtmp[1]*vecik[2];
                        tmpd[6] = dangtmp[2]*vecjk[0];
                        tmpd[7] = dangtmp[2]*vecjk[1];
                        tmpd[8] = dangtmp[2]*vecjk[2];

                        for (int a=0; a < 3; ++a) {
                            dsymf[ii][s*natoms*3 + nei_list_i[j*2 + 1]*3 + a] += tmpd[a] - tmpd[a + 6];
                            dsymf[ii][s*natoms*3 + nei_list_i[k*2 + 1]*3 + a] += tmpd[a + 3] + tmpd[a + 6];
                            dsymf[ii][s*natoms*3 + i*3 + a] -= tmpd[a] + tmpd[a + 3];

                            for (int b=0; b < 3; ++b){
                                dsymfa[ii][s*3*6 + b*6 + a]     += (tmpd[a]*lcoeff[b] + tmpd[a + 3]*lcoeff[b + 3])*cell[b][a]/vol;
                                dsymfa[ii][s*3*6 + b*6 + a + 3] += (tmpd[a]*lcoeff[b] + tmpd[a + 3]*lcoeff[b + 3])*cell[b][(a + 1)%3]/vol;
                            }
                        }
                    }
                    else continue;
                }
            }
        }
        
        delete[] nei_list_d;
        delete[] nei_list_i;
    }
    
    for (int i=0; i<natoms; i++) {
        delete[] bin_i[i];
    }
    delete[] bin_i;
    delete[] powtwo;
    return 0;
}

void PyInit_libsymf(void) { } // for windows
