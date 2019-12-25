#include <math.h>
/*
 Code for calculate symmetry function.
 This code is used for both Python and LAMMPS code
 */
const int IMPLEMENTED_TYPE[] = {2, 4, 5}; // Change this when you implement new symfunc type!

static inline double pow_int(const double &x, const double n) {
    double res,tmp;

    if (x == 0.0) return 0.0; // FIXME: abs(x) < epsilon
    int nn = (n > 0) ? n : -n;
    tmp = x;

    for (res = 1.0; nn != 0; nn >>= 1, tmp *= tmp)
        if (nn & 1) res *= tmp;

    return (n > 0) ? res : 1.0/res;
}

static inline double sigm(double x, double &deriv) {
    double expl = 1./(1.+exp(-x));
    deriv = expl*(1-expl);
    return expl;
}

static inline double tanh(double x, double &deriv) {
    double expl = 2./(1.+exp(-2.*x))-1;
    deriv = 1.-expl*expl;
    return expl;
}

static inline double relu(double x, double &deriv) {
    if (x > 0) {
        deriv = 1.;
        return x;
    } else {
        deriv = 0.;
        return 0;
    }
}

static inline double cutf(double frac) {
    // frac = dist / cutoff_dist
    if (frac >= 1.0) {
        return 0;
    } else { 
        return 0.5 * (1 + cos(M_PI*frac));
    }
}

static inline double dcutf(double dist, double cutd) {
    if (dist/cutd >= 1.0) {
        return 0;
    } else { 
        return -0.5 * M_PI * sin(M_PI*dist/cutd) / cutd;
    }
}

static inline double G2(double Rij, double *precal, double *par, double &deriv) {
    // par[0] = cutoff_dist
    // par[1] = eta
    // par[2] = R_s
    double tmp = Rij-par[2];
    double expl = exp(-par[1]*tmp*tmp);
    deriv = expl*(-2*par[1]*tmp*precal[0] + precal[1]);
    return expl*precal[0];
}

static inline double G4(double Rij, double Rik, double Rjk, double powtwo, \
          double *precal, double *par, double *deriv) {
    // cosv: cos(theta)
    // par[0] = cutoff_dist
    // par[1] = eta
    // par[2] = zeta
    // par[3] = lambda
    double expl = exp(-par[1]*precal[6]) * powtwo;
    double cosv = 1 + par[3]*precal[7];
    //double powcos = pow_int(cosv, par[2]-1);
    double powcos = pow(fabs(cosv), fabs(par[2]-1));

    deriv[0] = expl*powcos*precal[2]*precal[4] * \
               ((-2*par[1]*Rij*precal[0] + precal[1])*cosv + \
               par[2]*par[3]*precal[0]*precal[8]); // ij
    deriv[1] = expl*powcos*precal[0]*precal[4] * \
               ((-2*par[1]*Rik*precal[2] + precal[3])*cosv + \
               par[2]*par[3]*precal[2]*precal[9]); // ik
    deriv[2] = expl*powcos*precal[0]*precal[2] * \
               ((-2*par[1]*Rjk*precal[4] + precal[5])*cosv + \
               par[2]*par[3]*precal[4]*precal[10]); // jk

    return powcos*cosv * expl * precal[0] * precal[2] * precal[4];
}

static inline double G5(double Rij, double Rik, double powtwo, \
          double *precal, double *par, double *deriv) {
    // cosv: cos(theta)
    // par[0] = cutoff_dist
    // par[1] = eta
    // par[2] = zeta
    // par[3] = lambda
    double expl = exp(-par[1]*precal[11]) * powtwo;
    double cosv = 1 + par[3]*precal[7];
    //double powcos = pow_int(cosv, par[2]-1);
    double powcos = pow(fabs(cosv), fabs(par[2]-1));

    deriv[0] = expl*powcos*precal[2] * \
               ((-2*par[1]*Rij*precal[0] + precal[1])*cosv + \
               par[2]*par[3]*precal[0]*precal[8]); // ij
    deriv[1] = expl*powcos*precal[0] * \
               ((-2*par[1]*Rik*precal[2] + precal[3])*cosv + \
               par[2]*par[3]*precal[2]*precal[9]); // ik
    deriv[2] = expl*powcos*precal[0]*precal[2] * \
               (par[2]*par[3]*precal[10]); // jk

    return powcos*cosv * expl * precal[0] * precal[2];
}
