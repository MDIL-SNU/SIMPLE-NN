import cffi

ffibuilder = cffi.FFI()
ffibuilder.cdef(
    """int calculate_sf(double **, double **, double **,
                                    int *, int, int*, int,
                                    int**, double **, int,
                                    double**, double**, double**);"""
)
ffibuilder.set_source(
    "simple_nn.features.symmetry_function._libsymf",
    '#include "calculate_sf.h"',
    sources=[
        "simple_nn/features/symmetry_function/calculate_sf.cpp",
    ],
    source_extension=".cpp",
    include_dirs=["simple_nn/features/symmetry_function/"],
)

if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
