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

static inline void cutf2(const double dist, const double cutd, double& f, double& df, int slot) {
    static double f_[3], df_[3], dist_[3], cutd_[3];
    if (dist_[slot] == dist && cutd_[slot] == cutd) {
        f = f_[slot];
        df = df_[slot];
        return;
    }
    double frac = dist / cutd;
    if (frac >= 1.0) {
        f = 0.0;
        df = 0.0;
    } else {
        double cos, sin;
        sincos(M_PI*frac, &sin, &cos);
        f = 0.5 * (1 + cos);
        df = -0.5 * M_PI * sin / cutd;
    }
    dist_[slot] = dist;
    cutd_[slot] = cutd;
    f_[slot] = f;
    df_[slot] = df;
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

static inline double G2(const double Rij, const double* precal, const double* par, double &deriv) {
    // par[0] = R_c
    // par[1] = eta
    // par[2] = R_s
    double tmp = Rij-par[2];
    double expl = exp(-par[1]*tmp*tmp);
    deriv = expl*(-2*par[1]*tmp*precal[0] + precal[1]);
    return expl*precal[0];
}

static inline double G4(const double Rij, const double Rik, const double Rjk, const double powtwo, \
          const double* precal, const double* par, double *deriv, const bool powint) {
    // par[0] = R_c
    // par[1] = eta
    // par[2] = zeta
    // par[3] = lambda
    double expl = exp(-par[1]*precal[6]) * powtwo;
    double cosv = 1 + par[3]*precal[7];
    double powcos = powint ? pow_int(fabs(cosv), par[2]-1) : pow(fabs(cosv), fabs(par[2]-1));

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

static inline double G5(const double Rij, const double Rik, const double powtwo, \
          const double* precal, const double* par, double *deriv, const bool powint) {
    // par[0] = R_c
    // par[1] = eta
    // par[2] = zeta
    // par[3] = lambda
    double expl = exp(-par[1]*precal[11]) * powtwo;
    double cosv = 1 + par[3]*precal[7];
    double powcos = powint ? pow_int(fabs(cosv), par[2]-1) : pow(fabs(cosv), fabs(par[2]-1));

    deriv[0] = expl*powcos*precal[2] * \
               ((-2*par[1]*Rij*precal[0] + precal[1])*cosv + \
               par[2]*par[3]*precal[0]*precal[8]); // ij
    deriv[1] = expl*powcos*precal[0] * \
               ((-2*par[1]*Rik*precal[2] + precal[3])*cosv + \
               par[2]*par[3]*precal[2]*precal[9]); // ik
    deriv[2] = expl*powcos*precal[0]*precal[2] * \
               par[2]*par[3]*precal[10]; // jk

    return powcos*cosv * expl * precal[0] * precal[2];
}
