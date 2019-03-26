#include <math.h>

// TODO: check the scale
extern "C" void calculate_gdf(double** refs, int num_refs, double** targets, int num_targets, int num_features, double sigma, double* gdf) {
    /*
    C extern function for calculating GDF value [paper ref.]
    features: [# of points, # of features]
    gdf: [# of points]
    */
    double tmp_gdf, tmp_indi, tot_val;
    double epsilon = 1e-20;

    tot_val = 0;
    for (int i=0; i<num_targets; ++i) {
        if (sigma > 0.) {
            tmp_gdf = 0;
            for (int j=0; j<num_refs; ++j) {
                tmp_indi = 0;
                for (int k=0; k<num_features; ++k) {
                    tmp_indi += (targets[i][k] - refs[j][k]) * (targets[i][k] - refs[j][k]);
                }
                tmp_gdf += exp(-tmp_indi/sigma/sigma/2/num_features);
            }
            gdf[i] = 1.0 / tmp_gdf;
        }
        else {
            tmp_gdf = 100.; // dummy value, minimum distance
            for (int j=0; j<num_refs; ++j) {
                tmp_indi = 0.;
                for (int k=0; k<num_features; ++k) {
                    tmp_indi += (targets[i][k] - refs[j][k]) * (targets[i][k] - refs[j][k]);
                }
                tmp_indi = sqrt(tmp_indi/num_features);
                if ((tmp_gdf > tmp_indi) && (tmp_indi > epsilon)) {
                    tmp_gdf = tmp_indi;
                }
            }
            gdf[i] = tmp_gdf;
        }
    }
}

void PyInit_libgdf(void) { } // for windows
