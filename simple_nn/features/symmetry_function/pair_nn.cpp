/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: 
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pair_nn.h"
#include "symmetry_functions.h"
#include "atom.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "error.h"
#include "pointers.h"
#include "utils.h"

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;

#define MAXLINE 50000

/* ---------------------------------------------------------------------- */
// Constructor

PairNN::PairNN(LAMMPS *lmp) : Pair(lmp) {
  nelements = 0;
  elements = NULL;
  map = NULL;
  //manybody_flag = 1;
}

/* ---------------------------------------------------------------------- */
// Destructor

PairNN::~PairNN()
{
  if (elements)
    for (int i = 0; i < nelements; i++) delete [] elements[i];
  delete [] elements;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    delete [] map;
  }

  for (int i=0; i<nelements; i++) free_net(nets[i]);
  delete [] nets;
}

/* ---------------------------------------------------------------------- */

void PairNN::compute(int eflag, int vflag)
{
  int i,j,k,n,ii,jj,kk,tt,nn,inum,jnum;
  int itype,jtype,ktype,ielem,jelem,kelem;
  double xtmp,ytmp,ztmp,dradtmp,tmpc,tmpE;
  double dangtmp[3];
  double tmpd[9];
  double precal[12];
  // precal: cfij, dcfij, cfik, dcfik, cfjk, dcfjk, dist_square_sum,
  //         cosval, dcosval/dij, dcosval/dik, dcosval/djk
  double delij[3],delik[3],deljk[3],vecij[3],vecik[3],vecjk[3],lcoeff[9];
  double cell[3][3], cross[3][3], inv[3][3];
  double vol,Rij,Rik,Rjk,rRij,rRik,rRjk,cutij,cutik,cutjk;
  int *ilist,*jlist,*numneigh,**firstneigh;

  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  struct Symc *sym;
  int nsym;
  int *type = atom->type;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  // loop over neighbors of my atoms

  if (vflag_atom) {
    double boxxlo = domain->boxlo[0];
    double boxxhi = domain->boxhi[0];
    double boxylo = domain->boxlo[1];
    double boxyhi = domain->boxhi[1];
    double boxzlo = domain->boxlo[2];
    double boxzhi = domain->boxhi[2];
    double xy = domain->xy;
    double xz = domain->xz;
    double yz = domain->yz;

    cell[0][0] = boxxhi - boxxlo;
    cell[0][1] = 0;
    cell[0][2] = 0;
    cell[1][0] = xy;
    cell[1][1] = boxyhi - boxylo;
    cell[1][2] = 0;
    cell[2][0] = xz;
    cell[2][1] = yz;
    cell[2][2] = boxzhi - boxzlo;

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
    inv[0][0] = cross[0][0]/vol;
    inv[0][1] = cross[1][0]/vol;
    inv[0][2] = cross[2][0]/vol;
    inv[1][0] = cross[0][1]/vol;
    inv[1][1] = cross[1][1]/vol;
    inv[1][2] = cross[2][1]/vol;
    inv[2][0] = cross[0][2]/vol;
    inv[2][1] = cross[1][2]/vol;
    inv[2][2] = cross[2][2]/vol;
  }

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    ielem = map[itype];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    int numshort = 0;
    nsym = nets[ielem].nnode[0];

    double *symvec = new double[nsym]();
    double *dsymvec = new double[nsym]();
    double *tmpf = new double[nsym*(jnum+1)*3]();
    double *tmps = new double[nsym*3*6]();

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      //j &= NEIGHMASK; // What is this?

      delij[0] = x[j][0] - xtmp;
      delij[1] = x[j][1] - ytmp;
      delij[2] = x[j][2] - ztmp;
      Rij = delij[0]*delij[0] + delij[1]*delij[1] + delij[2]*delij[2];
      jtype = type[j];
      jelem = map[jtype];

      if (Rij < 0.0001 || Rij > cutsq[itype][jtype]) { continue; }

      rRij = sqrt(Rij);
      vecij[0] = delij[0]/rRij;
      vecij[1] = delij[1]/rRij;
      vecij[2] = delij[2]/rRij;
      
      if (vflag_atom) {
        lcoeff[0] = delij[0]*inv[0][0] + delij[1]*inv[1][0] + delij[2]*inv[2][0];
        lcoeff[1] = delij[0]*inv[0][1] + delij[1]*inv[1][1] + delij[2]*inv[2][1];
        lcoeff[2] = delij[0]*inv[0][2] + delij[1]*inv[1][2] + delij[2]*inv[2][2];
      }

      // calc radial symfunc
      for (tt=0; tt<nsym; tt++) {
        sym = &nets[ielem].slists[tt];
        if (rRij > sym->coefs[0]) continue;
        if (sym->atype[0] != jelem) continue;
        if (sym->stype == 2) {
          cutf2(rRij, sym->coefs[0], precal[0], precal[1], 0);

          symvec[tt] += G2(rRij, precal, sym->coefs, dradtmp);
          
          tmpd[0] = dradtmp*vecij[0];
          tmpd[1] = dradtmp*vecij[1];
          tmpd[2] = dradtmp*vecij[2];
          
          tmpf[tt*(jnum+1)*3 + jj*3]     += tmpd[0];
          tmpf[tt*(jnum+1)*3 + jj*3 + 1] += tmpd[1];
          tmpf[tt*(jnum+1)*3 + jj*3 + 2] += tmpd[2];
          
          tmpf[tt*(jnum+1)*3 + jnum*3]     -= tmpd[0];
          tmpf[tt*(jnum+1)*3 + jnum*3 + 1] -= tmpd[1];
          tmpf[tt*(jnum+1)*3 + jnum*3 + 2] -= tmpd[2];
          
          if (vflag_atom) {
            tmps[tt*3*6]      += tmpd[0]*lcoeff[0]*cell[0][0];
            tmps[tt*3*6 + 1]  += tmpd[1]*lcoeff[0]*cell[0][1];
            tmps[tt*3*6 + 2]  += tmpd[2]*lcoeff[0]*cell[0][2];
            tmps[tt*3*6 + 3]  += tmpd[0]*lcoeff[0]*cell[0][1];
            tmps[tt*3*6 + 4]  += tmpd[1]*lcoeff[0]*cell[0][2];
            tmps[tt*3*6 + 5]  += tmpd[2]*lcoeff[0]*cell[0][0];
            tmps[tt*3*6 + 6]  += tmpd[0]*lcoeff[1]*cell[1][0];
            tmps[tt*3*6 + 7]  += tmpd[1]*lcoeff[1]*cell[1][1];
            tmps[tt*3*6 + 8]  += tmpd[2]*lcoeff[1]*cell[1][2];
            tmps[tt*3*6 + 9]  += tmpd[0]*lcoeff[1]*cell[1][1];
            tmps[tt*3*6 + 10] += tmpd[1]*lcoeff[1]*cell[1][2];
            tmps[tt*3*6 + 11] += tmpd[2]*lcoeff[1]*cell[1][0];
            tmps[tt*3*6 + 12] += tmpd[0]*lcoeff[2]*cell[2][0];
            tmps[tt*3*6 + 13] += tmpd[1]*lcoeff[2]*cell[2][1];
            tmps[tt*3*6 + 14] += tmpd[2]*lcoeff[2]*cell[2][2];
            tmps[tt*3*6 + 15] += tmpd[0]*lcoeff[2]*cell[2][1];
            tmps[tt*3*6 + 16] += tmpd[1]*lcoeff[2]*cell[2][2];
            tmps[tt*3*6 + 17] += tmpd[2]*lcoeff[2]*cell[2][0];
          }
        }
        else continue;
      }

      if (rRij > max_rc_ang) continue;
      for (kk = jj+1; kk < jnum; kk++) {
        k = jlist[kk];
        //k &= NEIGHMASK;
        delik[0] = x[k][0] - xtmp;
        delik[1] = x[k][1] - ytmp;
        delik[2] = x[k][2] - ztmp;
        Rik = delik[0]*delik[0] + delik[1]*delik[1] + delik[2]*delik[2];
        if (Rik < 0.0001) continue;
        rRik = sqrt(Rik);
        if (rRik > max_rc_ang) continue;
        ktype = type[k];
        kelem = map[ktype];
        
        deljk[0] = x[k][0] - x[j][0];
        deljk[1] = x[k][1] - x[j][1];
        deljk[2] = x[k][2] - x[j][2];
        Rjk = deljk[0]*deljk[0] + deljk[1]*deljk[1] + deljk[2]*deljk[2];
        if (Rjk < 0.0001) continue;
        rRjk = sqrt(Rjk);

        vecik[0] = delik[0]/rRik;
        vecik[1] = delik[1]/rRik;
        vecik[2] = delik[2]/rRik;
     
        vecjk[0] = deljk[0]/rRjk;
        vecjk[1] = deljk[1]/rRjk;
        vecjk[2] = deljk[2]/rRjk;
       
        if (vflag_atom) {
          lcoeff[3] = delik[0]*inv[0][0] + delik[1]*inv[1][0] + delik[2]*inv[2][0];
          lcoeff[4] = delik[0]*inv[0][1] + delik[1]*inv[1][1] + delik[2]*inv[2][1];
          lcoeff[5] = delik[0]*inv[0][2] + delik[1]*inv[1][2] + delik[2]*inv[2][2];
         
          lcoeff[6] = deljk[0]*inv[0][0] + deljk[1]*inv[1][0] + deljk[2]*inv[2][0];
          lcoeff[7] = deljk[0]*inv[0][1] + deljk[1]*inv[1][1] + deljk[2]*inv[2][1];
          lcoeff[8] = deljk[0]*inv[0][2] + deljk[1]*inv[1][2] + deljk[2]*inv[2][2];
        }

        // Note that Rij = rRij * rRij.
        precal[7] = (Rij + Rik - Rjk)/2/rRij/rRik;
        // dcos(theta)/db = precal[8]
        // dcos(theta)/dc = precal[9]
        // dcos(theta)/da = precal[10]
        precal[8] = 0.5*(1/rRik + 1/Rij*(Rjk/rRik - rRik));
        precal[9] = 0.5*(1/rRij + 1/Rik*(Rjk/rRij - rRij));
        precal[10] = -rRjk/rRij/rRik;
        // Calculate precal[6] only when there is at least one G4.
        // (precal[6] is only used by G4.)
        if (nsf[4] > 0) {
          precal[6] = Rij+Rik+Rjk;
        }
        if (nsf[5] > 0) {
          precal[11] = Rij+Rik;
        }

        // calc angular symfunc
        for (tt=0; tt<nsym; tt++) {
          sym = &nets[ielem].slists[tt];
          if (rRik > sym->coefs[0]) continue;
          if (!((sym->atype[0] == jelem && sym->atype[1] == kelem) || \
                (sym->atype[0] == kelem && sym->atype[1] == jelem))) continue;
          if ((sym->stype) == 4) {
            if (rRjk > sym->coefs[0]) continue;
            cutf2(rRij, nets[ielem].slists[tt].coefs[0], precal[0], precal[1], 0);
            cutf2(rRik, nets[ielem].slists[tt].coefs[0], precal[2], precal[3], 1);
            cutf2(rRjk, nets[ielem].slists[tt].coefs[0], precal[4], precal[5], 2);

            symvec[tt] += G4(rRij, rRik, rRjk, nets[ielem].powtwo[tt], precal, sym->coefs, dangtmp, nets[ielem].powint[tt]);

            tmpd[0] = dangtmp[0]*vecij[0];
            tmpd[1] = dangtmp[0]*vecij[1];
            tmpd[2] = dangtmp[0]*vecij[2];
            tmpd[3] = dangtmp[1]*vecik[0];
            tmpd[4] = dangtmp[1]*vecik[1];
            tmpd[5] = dangtmp[1]*vecik[2];
            tmpd[6] = dangtmp[2]*vecjk[0];
            tmpd[7] = dangtmp[2]*vecjk[1];
            tmpd[8] = dangtmp[2]*vecjk[2];

            tmpf[tt*(jnum+1)*3 + jj*3]      += tmpd[0] - tmpd[6];
            tmpf[tt*(jnum+1)*3 + jj*3 + 1]  += tmpd[1] - tmpd[7];
            tmpf[tt*(jnum+1)*3 + jj*3 + 2]  += tmpd[2] - tmpd[8];

            tmpf[tt*(jnum+1)*3 + kk*3]      += tmpd[3] + tmpd[6];
            tmpf[tt*(jnum+1)*3 + kk*3 + 1]  += tmpd[4] + tmpd[7];
            tmpf[tt*(jnum+1)*3 + kk*3 + 2]  += tmpd[5] + tmpd[8];

            tmpf[tt*(jnum+1)*3 + jnum*3]     -= tmpd[0] + tmpd[3];
            tmpf[tt*(jnum+1)*3 + jnum*3 + 1] -= tmpd[1] + tmpd[4];
            tmpf[tt*(jnum+1)*3 + jnum*3 + 2] -= tmpd[2] + tmpd[5];

            if (vflag_atom) {
              tmps[tt*3*6]      += (tmpd[0]*lcoeff[0] + tmpd[3]*lcoeff[3] + tmpd[6]*lcoeff[6])*cell[0][0];
              tmps[tt*3*6 + 1]  += (tmpd[1]*lcoeff[0] + tmpd[4]*lcoeff[3] + tmpd[7]*lcoeff[6])*cell[0][1];
              tmps[tt*3*6 + 2]  += (tmpd[2]*lcoeff[0] + tmpd[5]*lcoeff[3] + tmpd[8]*lcoeff[6])*cell[0][2];
              tmps[tt*3*6 + 3]  += (tmpd[0]*lcoeff[0] + tmpd[3]*lcoeff[3] + tmpd[6]*lcoeff[6])*cell[0][1];
              tmps[tt*3*6 + 4]  += (tmpd[1]*lcoeff[0] + tmpd[4]*lcoeff[3] + tmpd[7]*lcoeff[6])*cell[0][2];
              tmps[tt*3*6 + 5]  += (tmpd[2]*lcoeff[0] + tmpd[5]*lcoeff[3] + tmpd[8]*lcoeff[6])*cell[0][0];
              tmps[tt*3*6 + 6]  += (tmpd[0]*lcoeff[1] + tmpd[3]*lcoeff[4] + tmpd[6]*lcoeff[7])*cell[1][0];
              tmps[tt*3*6 + 7]  += (tmpd[1]*lcoeff[1] + tmpd[4]*lcoeff[4] + tmpd[7]*lcoeff[7])*cell[1][1];
              tmps[tt*3*6 + 8]  += (tmpd[2]*lcoeff[1] + tmpd[5]*lcoeff[4] + tmpd[8]*lcoeff[7])*cell[1][2];
              tmps[tt*3*6 + 9]  += (tmpd[0]*lcoeff[1] + tmpd[3]*lcoeff[4] + tmpd[6]*lcoeff[7])*cell[1][1];
              tmps[tt*3*6 + 10] += (tmpd[1]*lcoeff[1] + tmpd[4]*lcoeff[4] + tmpd[7]*lcoeff[7])*cell[1][2];
              tmps[tt*3*6 + 11] += (tmpd[2]*lcoeff[1] + tmpd[5]*lcoeff[4] + tmpd[8]*lcoeff[7])*cell[1][0];
              tmps[tt*3*6 + 12] += (tmpd[0]*lcoeff[2] + tmpd[3]*lcoeff[5] + tmpd[6]*lcoeff[8])*cell[2][0];
              tmps[tt*3*6 + 13] += (tmpd[1]*lcoeff[2] + tmpd[4]*lcoeff[5] + tmpd[7]*lcoeff[8])*cell[2][1];
              tmps[tt*3*6 + 14] += (tmpd[2]*lcoeff[2] + tmpd[5]*lcoeff[5] + tmpd[8]*lcoeff[8])*cell[2][2];
              tmps[tt*3*6 + 15] += (tmpd[0]*lcoeff[2] + tmpd[3]*lcoeff[5] + tmpd[6]*lcoeff[8])*cell[2][1];
              tmps[tt*3*6 + 16] += (tmpd[1]*lcoeff[2] + tmpd[4]*lcoeff[5] + tmpd[7]*lcoeff[8])*cell[2][2];
              tmps[tt*3*6 + 17] += (tmpd[2]*lcoeff[2] + tmpd[5]*lcoeff[5] + tmpd[8]*lcoeff[8])*cell[2][0];
            }
          }
          else if ((sym->stype) == 5) {
            cutf2(rRij, nets[ielem].slists[tt].coefs[0], precal[0], precal[1], 0);
            cutf2(rRik, nets[ielem].slists[tt].coefs[0], precal[2], precal[3], 1);

            symvec[tt] += G5(rRij, rRik, nets[ielem].powtwo[tt], precal, sym->coefs, dangtmp, nets[ielem].powint[tt]);

            tmpd[0] = dangtmp[0]*vecij[0];
            tmpd[1] = dangtmp[0]*vecij[1];
            tmpd[2] = dangtmp[0]*vecij[2];
            tmpd[3] = dangtmp[1]*vecik[0];
            tmpd[4] = dangtmp[1]*vecik[1];
            tmpd[5] = dangtmp[1]*vecik[2];
            tmpd[6] = dangtmp[2]*vecjk[0];
            tmpd[7] = dangtmp[2]*vecjk[1];
            tmpd[8] = dangtmp[2]*vecjk[2];

            tmpf[tt*(jnum+1)*3 + jj*3]      += tmpd[0] - tmpd[6];
            tmpf[tt*(jnum+1)*3 + jj*3 + 1]  += tmpd[1] - tmpd[7];
            tmpf[tt*(jnum+1)*3 + jj*3 + 2]  += tmpd[2] - tmpd[8];

            tmpf[tt*(jnum+1)*3 + kk*3]      += tmpd[3] + tmpd[6];
            tmpf[tt*(jnum+1)*3 + kk*3 + 1]  += tmpd[4] + tmpd[7];
            tmpf[tt*(jnum+1)*3 + kk*3 + 2]  += tmpd[5] + tmpd[8];

            tmpf[tt*(jnum+1)*3 + jnum*3]     -= tmpd[0] + tmpd[3];
            tmpf[tt*(jnum+1)*3 + jnum*3 + 1] -= tmpd[1] + tmpd[4];
            tmpf[tt*(jnum+1)*3 + jnum*3 + 2] -= tmpd[2] + tmpd[5];

            if (vflag_atom) {
              tmps[tt*3*6]      += (tmpd[0]*lcoeff[0] + tmpd[3]*lcoeff[3] + tmpd[6]*lcoeff[6])*cell[0][0];
              tmps[tt*3*6 + 1]  += (tmpd[1]*lcoeff[0] + tmpd[4]*lcoeff[3] + tmpd[7]*lcoeff[6])*cell[0][1];
              tmps[tt*3*6 + 2]  += (tmpd[2]*lcoeff[0] + tmpd[5]*lcoeff[3] + tmpd[8]*lcoeff[6])*cell[0][2];
              tmps[tt*3*6 + 3]  += (tmpd[0]*lcoeff[0] + tmpd[3]*lcoeff[3] + tmpd[6]*lcoeff[6])*cell[0][1];
              tmps[tt*3*6 + 4]  += (tmpd[1]*lcoeff[0] + tmpd[4]*lcoeff[3] + tmpd[7]*lcoeff[6])*cell[0][2];
              tmps[tt*3*6 + 5]  += (tmpd[2]*lcoeff[0] + tmpd[5]*lcoeff[3] + tmpd[8]*lcoeff[6])*cell[0][0];
              tmps[tt*3*6 + 6]  += (tmpd[0]*lcoeff[1] + tmpd[3]*lcoeff[4] + tmpd[6]*lcoeff[7])*cell[1][0];
              tmps[tt*3*6 + 7]  += (tmpd[1]*lcoeff[1] + tmpd[4]*lcoeff[4] + tmpd[7]*lcoeff[7])*cell[1][1];
              tmps[tt*3*6 + 8]  += (tmpd[2]*lcoeff[1] + tmpd[5]*lcoeff[4] + tmpd[8]*lcoeff[7])*cell[1][2];
              tmps[tt*3*6 + 9]  += (tmpd[0]*lcoeff[1] + tmpd[3]*lcoeff[4] + tmpd[6]*lcoeff[7])*cell[1][1];
              tmps[tt*3*6 + 10] += (tmpd[1]*lcoeff[1] + tmpd[4]*lcoeff[4] + tmpd[7]*lcoeff[7])*cell[1][2];
              tmps[tt*3*6 + 11] += (tmpd[2]*lcoeff[1] + tmpd[5]*lcoeff[4] + tmpd[8]*lcoeff[7])*cell[1][0];
              tmps[tt*3*6 + 12] += (tmpd[0]*lcoeff[2] + tmpd[3]*lcoeff[5] + tmpd[6]*lcoeff[8])*cell[2][0];
              tmps[tt*3*6 + 13] += (tmpd[1]*lcoeff[2] + tmpd[4]*lcoeff[5] + tmpd[7]*lcoeff[8])*cell[2][1];
              tmps[tt*3*6 + 14] += (tmpd[2]*lcoeff[2] + tmpd[5]*lcoeff[5] + tmpd[8]*lcoeff[8])*cell[2][2];
              tmps[tt*3*6 + 15] += (tmpd[0]*lcoeff[2] + tmpd[3]*lcoeff[5] + tmpd[6]*lcoeff[8])*cell[2][1];
              tmps[tt*3*6 + 16] += (tmpd[1]*lcoeff[2] + tmpd[4]*lcoeff[5] + tmpd[7]*lcoeff[8])*cell[2][2];
              tmps[tt*3*6 + 17] += (tmpd[2]*lcoeff[2] + tmpd[5]*lcoeff[5] + tmpd[8]*lcoeff[8])*cell[2][0];
            }
          }
        }
      }
    }
    
    double *scale1 = new double[nsym];
    for (tt=0; tt<nsym; tt++) {
      scale1[tt] = nets[ielem].scale[1][tt];
      symvec[tt] = (symvec[tt] - nets[ielem].scale[0][tt])/scale1[tt];
    }

    tmpE = evalNet(symvec, dsymvec, nets[ielem]); // change E variable
    if (eflag_global) { eng_vdwl += tmpE; }
    if (eflag_atom) { eatom[i] += tmpE; }

    // update force
    for (tt=0; tt<nsym; tt++) {
      tmpc = dsymvec[tt]/scale1[tt];
      for (nn = 0; nn < jnum; nn++) {
        n = jlist[nn];

        f[n][0] -= tmpf[tt*(jnum+1)*3 + nn*3 + 0]*tmpc;
        f[n][1] -= tmpf[tt*(jnum+1)*3 + nn*3 + 1]*tmpc;
        f[n][2] -= tmpf[tt*(jnum+1)*3 + nn*3 + 2]*tmpc;
      }
      f[i][0] -= tmpf[tt*(jnum+1)*3 + jnum*3 + 0]*tmpc;
      f[i][1] -= tmpf[tt*(jnum+1)*3 + jnum*3 + 1]*tmpc;
      f[i][2] -= tmpf[tt*(jnum+1)*3 + jnum*3 + 2]*tmpc;

      if (vflag_atom) {
        // Order : xx yy zz zy xz yz
        vatom[i][0] -= tmps[tt*3*6]*tmpc;
        vatom[i][1] -= tmps[tt*3*6 + 1]*tmpc;
        vatom[i][2] -= tmps[tt*3*6 + 2]*tmpc;
        vatom[i][3] -= tmps[tt*3*6 + 3]*tmpc;
        vatom[i][5] -= tmps[tt*3*6 + 4]*tmpc;
        vatom[i][4] -= tmps[tt*3*6 + 5]*tmpc;
        vatom[i][0] -= tmps[tt*3*6 + 6]*tmpc;
        vatom[i][1] -= tmps[tt*3*6 + 7]*tmpc;
        vatom[i][2] -= tmps[tt*3*6 + 8]*tmpc;
        vatom[i][3] -= tmps[tt*3*6 + 9]*tmpc;
        vatom[i][5] -= tmps[tt*3*6 + 10]*tmpc;
        vatom[i][4] -= tmps[tt*3*6 + 11]*tmpc;
        vatom[i][0] -= tmps[tt*3*6 + 12]*tmpc;
        vatom[i][1] -= tmps[tt*3*6 + 13]*tmpc;
        vatom[i][2] -= tmps[tt*3*6 + 14]*tmpc;
        vatom[i][3] -= tmps[tt*3*6 + 15]*tmpc;
        vatom[i][5] -= tmps[tt*3*6 + 16]*tmpc;
        vatom[i][4] -= tmps[tt*3*6 + 17]*tmpc;
      }
    }

    delete [] symvec;
    delete [] dsymvec;
    delete [] tmpf;
    delete [] tmps;
    delete [] scale1;
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairNN::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  map = new int[n+1];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairNN::settings(int narg, char **arg)
{
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairNN::coeff(int narg, char **arg)
{
  int i,j,n;

  if (narg != 3 + atom->ntypes)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  // insure I,J args are * *

  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // read args that map atom types to elements in potential file
  // map[i] = which element the Ith atom type is, -1 if NULL
  // nelements = # of unique elements
  // elements = list of element names

  if (elements) {
    for (i = 0; i < nelements; i++) delete [] elements[i];
    delete [] elements;
  }

  elements = new char*[atom->ntypes];
  for (i = 0; i < atom->ntypes; i++) elements[i] = NULL;

  nelements = 0;
  for (i = 3; i < narg; i++) {
    if (strcmp(arg[i],"NULL") == 0) {
      map[i-2] = -1;
      continue;
    }
    for (j = 0; j < nelements; j++)
      if (strcmp(arg[i],elements[j]) == 0) break;
    map[i-2] = j;
    if (j == nelements) {
      n = strlen(arg[i]) + 1;
      elements[j] = new char[n];
      strcpy(elements[j],arg[i]);
      nelements++;
    }
  }

  nets = new Net[nelements+1]; // extra one is used for reading irrelevant elements.

  // read potential file and initialize potential parameters
  read_file(arg[2]);

  // clear setflag since coeff() called once with I,J = * *
  n = atom->ntypes;
  for (i = 1; i <= n; i++)
    for (j = i; j <= n; j++)
      setflag[i][j] = 0;

  // set setflag i,j for type pairs where both are mapped to elements
  int count = 0;
  for (i = 1; i <= n; i++)
    for (j = i; j <= n; j++)
      if (map[i] >= 0 && map[j] >= 0) {
        setflag[i][j] = 1;
        count++;
      }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

void PairNN::read_file(char *fname) {
  int i,j,k;
  FILE *fp;
  if (comm->me == 0) {
    //read file
    fp = fopen(fname, "r");
    if (fp == NULL) {
      char str[128];
      sprintf(str,"Cannot open NN potential file %s",fname);
      error->one(FLERR,str);
    }
  }

  int n,nwords,nsym,nlayer,isym,iscale,inode,ilayer,t_wb;
  char line[MAXLINE], *ptr, *tstr;
  int eof = 0;
  int stats = 0;
  int nnet = nelements;
  int max_sym_line = 6;
  char **p_elem = new char*[nelements];
  int valid_count = 0;
  bool valid = false;
  cutmax = 0;
  max_rc_ang = 0.0;
  for (int i=0; i<5+1; i++) {
    nsf[i] = 0;
  }

  while (1) {
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fp);
      if (ptr == NULL) {
        eof = 1;
        if (stats != 1) error->one(FLERR,"insufficient potential");
        fclose(fp);
      } else n = strlen(line) + 1;
    }
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(&n,1,MPI_INT,0,world);
    MPI_Bcast(line,n,MPI_CHAR,0,world);

    // strip comment, skip line if blank
    if ((ptr = strchr(line,'#'))) *ptr = '\0';
    nwords = utils::count_words(line);
    if (nwords == 0) continue;

    // get all potential parameters
    if (stats == 0) { // initialization
      // FIXME: p_elem usage?
      p_elem[0] = strtok(line," \t\n\r\f");
      for (i=1; i<nelements; i++) {
        p_elem[i] = strtok(NULL," \t\n\r\f");
      }
      stats = 1;
    } else if (stats == 1) { // potential element setting
      tstr = strtok(line," \t\n\r\f");
      char *t_elem = strtok(NULL," \t\n\r\f");
      double t_cut = atof(strtok(NULL," \t\n\r\f"));
      if (t_cut > cutmax) cutmax = t_cut;
      nnet = nelements;
      for (i=0; i<nelements; i++) {
        if (strcmp(t_elem,elements[i]) == 0) {
          nnet = i;
          break;
        }
      }
      if (nnet == nelements) {
        if (valid) {
          free_net(nets[nnet]);
        }
        valid = true;
      }
      else {
        valid_count++;
      }
      stats = 2;
      // cutoff setting
      for (i=1; i<=atom->ntypes; i++) {
        if (map[i] == nnet) {
          for (j=1; j<=atom->ntypes; j++) {
            cutsq[i][j] = t_cut*t_cut;
          }
        }
      }
    } else if (stats == 2) { // symfunc number setting
      tstr = strtok(line," \t\n\r\f");
      if (strncmp(tstr,"SYM",3) != 0)
        error->one(FLERR,"potential file error: missing info(# of symfunc)");
      nsym = atoi(strtok(NULL," \t\n\r\f"));
      nets[nnet].slists = new Symc[nsym]();
      nets[nnet].scale = new double*[2];
      for (i=0; i<2; ++i) {
        nets[nnet].scale[i] = new double[nsym];
      }
      stats = 3;
      isym = 0;
    } else if (stats == 3) { // read symfunc parameters
      nets[nnet].slists[isym].stype = atoi(strtok(line," \t\n\r\f"));
      nets[nnet].slists[isym].coefs[0] = atof(strtok(NULL," \t\n\r\f"));
      nets[nnet].slists[isym].coefs[1] = atof(strtok(NULL," \t\n\r\f"));
      nets[nnet].slists[isym].coefs[2] = atof(strtok(NULL," \t\n\r\f"));
      nets[nnet].slists[isym].coefs[3] = atof(strtok(NULL," \t\n\r\f"));

      tstr = strtok(NULL," \t\n\r\f");
      nets[nnet].slists[isym].atype[0] = nelements;
      for (i=0; i<nelements; i++) {
        if (strcmp(tstr, elements[i]) == 0) {
          nets[nnet].slists[isym].atype[0] = i;
          break;
        }
      }
      // In this code, SF type >= 4 means that it is angular function.
      if (nets[nnet].slists[isym].stype >= 4) {
        // Find maximum cutoff distance among angular functions.
        max_rc_ang = max(max_rc_ang, nets[nnet].slists[isym].coefs[0]);
        nsf[nets[nnet].slists[isym].stype] += 1;
        tstr = strtok(NULL," \t\n\r\f");
        nets[nnet].slists[isym].atype[1] = nelements;
        for (i=0; i<nelements; i++) {
          if (strcmp(tstr,elements[i]) == 0) {
            nets[nnet].slists[isym].atype[1] = i;
            break;
          }
        }
      }
    
      // Check for not implemented symfunc type.
      bool implemented = false;
      for (i=0; i < sizeof(IMPLEMENTED_TYPE) / sizeof(IMPLEMENTED_TYPE[0]); i++) {
        if ((nets[nnet].slists[isym].stype) == IMPLEMENTED_TYPE[i]) {
          implemented = true;
          break;
        }
      }
      if (!implemented) error->all(FLERR, "Not implemented symmetry function type!");

      if (nets[nnet].slists[isym].stype == 4 || nets[nnet].slists[isym].stype == 5) {
        if (nets[nnet].slists[isym].coefs[2] < 1.0) {
          error->all(FLERR, "Zeta in G4/G5 must be greater or equal to 1.0!");
        }
      }
    
      isym++;
      if (isym == nsym) {
        stats = 4;
        iscale = 0;
      }
    } else if (stats == 4) { // scale
      tstr = strtok(line," \t\n\r\f");
      //double *tmp_scale[nsym];
      for (i=0; i<nsym; i++) {
        //tmp_scale[i] = atof(strtok(NULL," \t\n\r\f"));
        nets[nnet].scale[iscale][i] = atof(strtok(NULL," \t\n\r\f"));
      }
      //nets[nnet].scale.push_back(tmp_scale);
      iscale++;
      if (iscale == 2) stats = 5;
    } else if (stats == 5) { // network number setting
      //nets[nnet].nnode.push_back(nsym);
      tstr = strtok(line," \t\n\r\f");
      nlayer = atoi(strtok(NULL," \t\n\r\f"));
      nlayer += 1;
      nets[nnet].nlayer = nlayer + 1;

      nets[nnet].nnode = new int[nets[nnet].nlayer];
      nets[nnet].nnode[0] = nsym;
      ilayer = 1;

      while ((tstr = strtok(NULL," \t\n\r\f"))) {
        nets[nnet].nnode[ilayer] = atoi(tstr);
        ilayer++;
      }
      nets[nnet].nodes = new double*[nlayer];
      nets[nnet].dnodes = new double*[nlayer];
      nets[nnet].bnodes = new double*[nlayer];
      for (i=0; i<nlayer; ++i) {
        nets[nnet].nodes[i] = new double[nets[nnet].nnode[i+1]];
        nets[nnet].dnodes[i] = new double[nets[nnet].nnode[i+1]];
        nets[nnet].bnodes[i] = new double[nets[nnet].nnode[i+1]];
      }
      nets[nnet].acti = new int[nlayer];

      nets[nnet].weights = new double*[nlayer];
      nets[nnet].bias = new double*[nlayer];
      for (i=0; i<nlayer; ++i) {
        nets[nnet].weights[i] = new double[nets[nnet].nnode[i]*nets[nnet].nnode[i+1]];
        nets[nnet].bias[i] = new double[nets[nnet].nnode[i+1]];
      }

      stats = 6;
      ilayer = 0;
    } else if (stats == 6) { // layer setting
      tstr = strtok(line," \t\n\r\f");
      tstr = strtok(NULL," \t\n\r\f");
      tstr = strtok(NULL," \t\n\r\f");
      if (strncmp(tstr, "linear", 6) == 0) nets[nnet].acti[ilayer] = 0;
      else if (strncmp(tstr, "sigmoid", 7) == 0) nets[nnet].acti[ilayer] = 1;
      else if (strncmp(tstr, "tanh", 4) == 0) nets[nnet].acti[ilayer] = 2;
      else if (strncmp(tstr, "relu", 4) == 0) nets[nnet].acti[ilayer] = 3;
      else nets[nnet].acti[ilayer] = 4;
      inode = 0;
      stats = 7;
      t_wb = 0;

    } else if (stats == 7) { // weights setting
      if (t_wb == 0) { // weights
        tstr = strtok(line," \t\n\r\f");
        for (i=0; i<nets[nnet].nnode[ilayer]; i++) {
          nets[nnet].weights[ilayer][inode*nets[nnet].nnode[ilayer] + i] = atof(strtok(NULL," \t\n\r\f"));
        }
        t_wb = 1;
      } else if (t_wb == 1) { // bias
        tstr = strtok(line," \t\n\r\f");
        nets[nnet].bias[ilayer][inode] = atof(strtok(NULL," \t\n\r\f"));
        t_wb = 0;
        inode++;
      }

      if (inode == nets[nnet].nnode[ilayer+1]) {
        ilayer++;
        stats = 6;
      }
      if (ilayer == nlayer) {
        if (nnet == nelements) free_net(nets[nnet]);
        stats = 1;
      }
    }
  }
  if (valid_count == 0) error->one(FLERR,"potential file error: invalid elements");

  delete [] p_elem;

  // pre-calculate some constants for symmetry functions.
  for (int i=0; i<nelements; i++) {
    int nsym = nets[i].nnode[0];
    nets[i].powtwo = new double[nsym];
    nets[i].powint = new bool[nsym];
    for (int tt=0; tt<nsym; tt++) {
      nets[i].powtwo[tt] = 0.0;
      nets[i].powint[tt] = false;

      if (nets[i].slists[tt].stype == 4 || \
          nets[i].slists[tt].stype == 5) {
        if (nets[i].slists[tt].coefs[2] < 1.0) {
          error->all(FLERR, "Zeta in G4/G5 must be greater or equal to 1.0!");
        }
        nets[i].powtwo[tt] = pow(2, 1-nets[i].slists[tt].coefs[2]);
        // powint indicates whether zeta is (almost) integer so that we can treat it as integer and use pow_int.
        // This is used because pow_int is much faster than pow.
        nets[i].powint[tt] = (nets[i].slists[tt].coefs[2] - int(nets[i].slists[tt].coefs[2])) < 1e-6;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairNN::init_style() {
    int irequest;

    irequest = neighbor->request(this,instance_me);
    neighbor->requests[irequest]->half = 0;
    neighbor->requests[irequest]->full = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairNN::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");
  return cutmax;
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairNN::write_restart(FILE *fp) {}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairNN::read_restart(FILE *fp) {}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairNN::write_restart_settings(FILE *fp) {}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairNN::read_restart_settings(FILE *fp) {}

/* ---------------------------------------------------------------------- */

double PairNN::single(int i, int j, int itype, int jtype, double rsq,
                         double factor_coul, double factor_lj,
                         double &fforce)
{
  if (comm->me == 0) printf("single run\n");
  return factor_lj;
}

/* ----------------------------------------------------------------------
   free the Nets struct
------------------------------------------------------------------------- */

void PairNN::free_net(Net &net) {
  for (int i=0; i<(net.nlayer-1); i++) {
    delete [] net.nodes[i];
    delete [] net.dnodes[i];
    delete [] net.bnodes[i];
    delete [] net.weights[i];
    delete [] net.bias[i];
  }
  delete [] net.nodes;
  delete [] net.dnodes;
  delete [] net.bnodes;
  delete [] net.weights;
  delete [] net.bias;

  for (int i=0; i<2; i++) {
    delete [] net.scale[i];
  }
  delete [] net.scale;

  delete [] net.acti;
  delete [] net.nnode;

  delete [] net.slists;
  delete [] net.powtwo;
  delete [] net.powint;
}


/* ---------------------------------------------------------------------- */

double PairNN::evalNet(const double* inpv, double *outv, Net &net){
// return energy and modify dE_dG
  int nl = net.nlayer;
  //int nl = (sizeof(net.nnode)) / sizeof(net.nnode[0]);

  // forwardprop
  // input - hidden1
  for (int i=0; i<net.nnode[1]; i++) {
    net.nodes[0][i] = 0;
    for (int j=0; j<net.nnode[0]; j++) {
      net.nodes[0][i] += net.weights[0][i*net.nnode[0]+j] * inpv[j];
    }
    if (net.acti[0] == 1) {
      net.nodes[0][i] = sigm(net.nodes[0][i] + net.bias[0][i], net.dnodes[0][i]);
    } 
    else if (net.acti[0] == 2) {
      net.nodes[0][i] = tanh(net.nodes[0][i] + net.bias[0][i], net.dnodes[0][i]);
    }
    else if (net.acti[0] == 3) {
      net.nodes[0][i] = relu(net.nodes[0][i] + net.bias[0][i], net.dnodes[0][i]);
    }
    else {
      net.nodes[0][i] += net.bias[0][i];
      net.dnodes[0][i] = 1;
    }
  }

  // hidden~output
  if (nl > 2) {
    for (int l=1; l<nl-1; l++) {
      for (int i=0; i<net.nnode[l+1]; i++) {
        net.nodes[l][i] = 0;
        for (int j=0; j<net.nnode[l]; j++) {
          net.nodes[l][i] += net.weights[l][i*net.nnode[l]+j] * net.nodes[l-1][j];
        }
        if (net.acti[l] == 1) {
          net.nodes[l][i] = sigm(net.nodes[l][i] + net.bias[l][i], net.dnodes[l][i]);
        } 
        else if (net.acti[l] == 2) {
          net.nodes[l][i] = tanh(net.nodes[l][i] + net.bias[l][i], net.dnodes[l][i]);
        }
        else if (net.acti[l] == 3) {
          net.nodes[l][i] = relu(net.nodes[l][i] + net.bias[l][i], net.dnodes[l][i]);
        }
        else {
          net.nodes[l][i] += net.bias[l][i];
          net.dnodes[l][i] = 1;
        }
      }
    }
  }

  // backwardprop
  // output layer dnode initialized to 1.
  for (int i=0; i<net.nnode[nl-1]; i++) {
    if (net.acti[nl-2] == 0) {
        net.bnodes[nl-2][i] = 1;
    }
    else {
        net.bnodes[nl-2][i] = net.dnodes[nl-2][i];
    }
  }

  if (nl > 2) {
    for (int l=nl-2; l>0; l--) {
      for (int i=0; i<net.nnode[l]; i++) {
        net.bnodes[l-1][i] = 0;
        for (int j=0; j<net.nnode[l+1]; j++) {
          net.bnodes[l-1][i] += net.weights[l][j*net.nnode[l]+i] * net.bnodes[l][j];
        }
        if (net.acti[l-1] >= 1) {
          net.bnodes[l-1][i] *= net.dnodes[l-1][i];
        }
      }
    }
  }

  for (int i=0; i<net.nnode[0]; i++) {
    outv[i] = 0;
    for (int j=0; j<net.nnode[1]; j++) {
      outv[i] += net.weights[0][j*net.nnode[0]+i] * net.bnodes[0][j];
    }
  }

  return net.nodes[nl-2][0]; // atomic energy
}
