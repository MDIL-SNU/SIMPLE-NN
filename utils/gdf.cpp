#include <math.h>

extern "C" void calculate_gdf(double** features, int num_points, int num_features, double sigma, double* gdf) {
    /*
    C extern function for calculating GDF value [paper ref.]
    features: [# of points, # of features]
    gdf: [# of points]
    */
    double tmp_gdf, tmp_indi;

    for (int i=0; i<num_points; ++i) {
        tmp_gdf = 0;
        for (int j=0; j<num_points; ++j) {
            tmp_indi = 0;
            for (int k=0; k<num_features; ++k) {
                tmp_indi += (features[i][k] - features[j][k]) * (features[i][k] - features[j][k]);
            }
            tmp_indi = exp(-tmp_gdf/sigma/sigma/2/num_features);
        }
        tmp_gdf += tmp_indi;
        gdf[i] = tmp_gdf / num_points;
        // TODO: scaling?
    }
}