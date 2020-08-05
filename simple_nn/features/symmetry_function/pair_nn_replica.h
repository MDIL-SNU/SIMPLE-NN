/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(nn/r,PairREPLICA)

#else

#ifndef LMP_PAIR_REPLICA
#define LMP_PAIR_REPLICA
#define _USE_MATH_DEFINES

#include "pair.h"
#include "math.h"
#include <iostream>
using namespace std;

namespace LAMMPS_NS {

class PairREPLICA : public Pair {
 public:
  PairREPLICA(class LAMMPS *);
  virtual ~PairREPLICA();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  double single(int, int, int, int, double, double, double, double &);

 protected:
// have to change
  struct Symc {
    int stype; // symmetry function type
    double coefs[4]; // symmetry function coefficients
    int atype[2]; // related atom type(tmp)
  };

  struct Net {
    // do not include input layer and dE/dG
    char *elem; // elements
    int *nnode; // number of nodes
    int nlayer; // number of layers
    int *acti; // activation function types
    double **weights; // weights
    double **bias; // bias
    double **nodes; // nodes
    double **dnodes; // nodes derivative
    double **bnodes; // nodes for backpropagation
    double **scale; // scale
    Symc *slists; // symmetry function related parameters
    double *powtwo; // power of two
    bool *powint;
  };

  Net *nets; // network parameters
  char **elements; // names of unique elements
  int nelements;   // # of unique elements
  int *map;        // mapping from atom types to elements
  double cutmax;
  double max_rc_ang;
  int nsf[5+1];   // number of symmetry functions with type N.
  int npot;       // number of potential

  virtual void allocate();
  virtual void read_file(char *, int);
  void free_net(Net &);
  double evalNet(const double *, double *, Net &);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

*/
