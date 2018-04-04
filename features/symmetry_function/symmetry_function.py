from __future__ import print_function
import os
from mpi4py import MPI
import numpy as np
from six.moves import cPickle as pickle
from ase import io
from cffi import FFI

def _gen_2Darray_for_ffi(arr, ffi, dtype=np.float64):
    # double precision with C style order
    arr = np.asarray(arr, dtype=np.float64, order='C') 
    shape = arr.shape
    arr_p = ffi.new("double *[%d]" % shape[0])
    for i in range(shape[0]):
        arr_p[i] = ffi.cast("double *", arr[i].ctypes.data)
    return arr_p

def _read_params(filename):
    params = list()
    with open(filename, 'r') as fil:
        for line in fil:
            params += map(float, line.split())

    params = np.array(params)
    return params

def feature_generator(structure_list, param_list):
    # get directory info for each structures and parameter list of symmetry functions

    # TODO: library read using ffi    
    ffi = FFI()
    # FIXME: add symmetry function vector and symmetry function derivative vector to input
    ffi.cdef("""void calculate_sf(double **, double **, double **, int *, double **, int, int);""")
    lib = ffi.dlopen(os.path.join(os.path.dirname(os.path.realpath(__file__)) + "/libsymf.so"))

    # FIXME: add test param_list to check the symmetry funciton calculation
    # parameter list
    # [symmetry function types, atom type 1, atom type 2, cutoff, parameter 1, 2, ...]
    #params = _read_params(param_list)
    params = np.array([[5.0, 3.0]])
    params_p = _gen_2Darray_for_ffi(params, ffi)

    # FIXME: take directory list and CONTCAR/XDATCAR info
    structure_list = [os.path.join(os.path.dirname(os.path.realpath(__file__)) + "/test/CONTCAR")]

    for item in structure_list:
        atoms = io.read(item)
        
        cart_p  = _gen_2Darray_for_ffi(atoms.positions, ffi)
        scale_p = _gen_2Darray_for_ffi(atoms.get_scaled_positions(), ffi)
        cell_p  = _gen_2Darray_for_ffi(atoms.cell, ffi)
        
        atom_i = np.array(atoms.get_chemical_symbols())
        for j,jtem in enumerate(set(atom_i)):
            atom_i[atom_i==jtem] = j # FIXME: index starting number 0 or 1?
        atom_i = atom_i.astype(np.int)
        atom_i_p = ffi.cast("int *", atom_i.ctypes.data)

        atom_num = len(atoms.positions)
        param_num = len(params)

        print("== check for C extension code ==\n")
        lib.calculate_sf(cell_p, cart_p, scale_p, atom_i_p, params_p, atom_num, param_num)

        #res = dict()
        #res['x'] = np.zeros([atom_num, param_num]).astype(np.float64)
        #res['dx'] = np.zeros([atom_num, atom_num, param_num, 3]).astype(np.float64)
        #res['params'] = params

        #x_p = _gen_2Darray_for_ffi(res['x'])
        #dx_p = _gen_2Darray_for_ffi(res['dx']) # TODO: change the dimension of res['dx']

        #feature_dict = calculate_feature(item)




        #pickle.dump(feature_dict, _name_, pickle.HIGHST_PROTOCOL)  # TODO: directory setting?


    return 0

"""
def calculate_feature(structure):
    comm = MPI.COMM_WORLD
    atoms = io.read(structure)  # TODO: add structure file information

    feature_dict = dict()
    feature_dict['sf'], feature_dict['dsf'] = _c_extension_for_calculating_sf_(atoms, comm)
    # TODO: make c extension

    return feature_dict
"""

if __name__ == "__main__":
    # for test
    feature_generator('as', 'as')
