#include <math.h>

// TODO: check the scale
extern "C" void calculate_gdf(double** features, int num_points, int num_features, double sigma, double* gdf) {
    /*
    C extern function for calculating GDF value [paper ref.]
    features: [# of points, # of features]
    gdf: [# of points]
    */
    double tmp_gdf, tmp_indi, tot_val;

    tot_val = 0;
    for (int i=0; i<num_points; ++i) {
        tmp_gdf = 0;
        for (int j=0; j<num_points; ++j) {
            if (i==j) continue;

            tmp_indi = 0;
            for (int k=0; k<num_features; ++k) {
                tmp_indi += (features[i][k] - features[j][k]) * (features[i][k] - features[j][k]);
            }
            tmp_gdf += exp(-tmp_indi/sigma/sigma/2/num_features);
        }

        // gdf[i] = tmp_gdf / num_points;
        gdf[i] = num_points / tmp_gdf;

        tot_val += gdf[i];
    }

    tot_val = tot_val / num_points;
    for (int i=0; i<num_points; ++i) {
        gdf[i] = gdf[i] / tot_val;
    }
}