import cffi

ffibuilder = cffi.FFI()
ffibuilder.cdef(
    """void calculate_gdf(double **, int, double **, int, int, double, double *);"""
)
ffibuilder.set_source(
    "simple_nn.utils._libgdf",
    '#include "gdf.h"',
    sources=["simple_nn/utils/gdf.cpp"],
    source_extension=".cpp",
    include_dirs=["simple_nn/utils/"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
