#include <math.h>
/*
 Code for calculate symmetry function.
 This code is used for both Python and LAMMPS code
 */
const int IMPLEMENTED_TYPE[] = {2, 4, 5, 6}; // Change this when you implement new symfunc type!

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
               ((-2*par[1]*Rjk*precal[4] + precal[5])*cosv - \
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
               -par[2]*par[3]*precal[10]; // jk

    return powcos*cosv * expl * precal[0] * precal[2];
}

// Modified angular symmetry function from ANI-1.
// J. S. Smith et al., Chem. Sci., 2017, 8, 3192.
static inline double G6(const double Rij, const double Rik, const double Rjk, const double powtwo, \
          const double sin_ts, const double cos_ts, const double *precal, const double *par, \
          double *deriv, bool powint) {
    // par[0] = R_c
    // par[1] = eta
    // par[2] = zeta
    // par[3] = R_s
    // par[4] = theta_s
    // precal[12] = sin(theta)
    // precal[13] = a*a - b*b - c*c
    // precal[14] = sqrt((a+b-c)*(a-b+c))
    // precal[15] = sqrt((-a+b+c)*(a+b+c))
    // precal[16] = 1 / (4*b*b*c*c*sqrt(b*c))
    double expl = exp(-par[1]*(0.5 * (Rij + Rik) - par[3]) * (0.5 * (Rij + Rik) - par[3])) * powtwo;
    double cosv = 1 + cos_ts * precal[7] + sin_ts * precal[12];
    double powcos = powint ? pow_int(fabs(cosv), par[2]-1) : pow(fabs(cosv), fabs(par[2]-1));

    double dsindx2 = (Rij - Rik) * precal[15] * precal[13] * precal[16];
    double dsindy2 = (Rij + Rik) * precal[14] * precal[13] * precal[16];
    double dcosdx2 = -(Rij - Rik) * precal[14] * precal[15] * precal[15] * precal[16];
    double dcosdy2 = (Rij + Rik) * precal[15] * precal[14] * precal[14] * precal[16];
    double cos_theta_2 = precal[15] / (2 * Rij * Rik);
    double sin_theta_2 = precal[14] / (2 * Rij * Rik);

    deriv[0] = sin_theta_2 * powcos*cosv * expl * (-precal[1] * precal[2] + precal[0] * precal[3]);
    deriv[1] = cos_theta_2 * powcos*cosv * expl * (-par[1] * ((Rij + Rik) - 2 * par[3]) * precal[0] * precal[2] + \
                                     precal[1] * precal[2] + precal[0] * precal[3]);
    deriv[0] += powcos * expl * precal[0] * precal[2] * par[2] * (cos_ts * dcosdx2 + sin_ts * dsindx2);
    deriv[1] += powcos * expl * precal[0] * precal[2] * par[2] * (cos_ts * dcosdy2 + sin_ts * dsindy2);
    deriv[2] = 0;

    return powcos*cosv * expl * precal[0] * precal[2];
}
