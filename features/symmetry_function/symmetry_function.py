from __future__ import print_function
import os
from mpi4py import MPI
import numpy as np
from six.moves import cPickle as pickle
from ase import io
from cffi import FFI

def _gen_2Darray_for_ffi(arr, ffi, dtype=np.float64, cdata="double"):
    # double precision with C style order
    arr = np.asarray(arr, dtype=dtype, order='C') 
    shape = arr.shape
    arr_p = ffi.new(cdata + " *[%d]" % shape[0])
    for i in range(shape[0]):
        arr_p[i] = ffi.cast(cdata + " *", arr[i].ctypes.data)
    return arr_p

def _read_params(filename):
    params_i = list()
    params_d = list()
    with open(filename, 'r') as fil:
        for line in fil:
            tmp = line.split()
            params_i += [list(map(int,   tmp[:3]))]
            params_d += [list(map(float, tmp[3:]))]

    return params_i, params_d

def feature_generator(structure_list, param_list):
    # get directory info for each structures and parameter list of symmetry functions

    # TODO: parallelize using mpi4py
    # TODO: library read using ffi    
    ffi = FFI()
    # FIXME: add symmetry function vector and symmetry function derivative vector to input
    ffi.cdef("""void calculate_sf(double **, double **, double **, int *, int **, double **, int, int, double **, double **);""")
    lib = ffi.dlopen(os.path.join(os.path.dirname(os.path.realpath(__file__)) + "/libsymf.so"))

    # FIXME: add test param_list to check the symmetry funciton calculation
    # parameter list
    # [symmetry function types, atom type 1, atom type 2, cutoff, parameter 1, 2, ...]
    params_i, params_d = _read_params(os.path.join(os.path.dirname(os.path.realpath(__file__)) + "/test/inp_fsymf"))
    #params = np.array([[5.0, 3.0]])
    params_ip = _gen_2Darray_for_ffi(params_i, ffi, np.int, "int")
    print(params_ip[0], params_ip[1])
    params_dp = _gen_2Darray_for_ffi(params_d, ffi, np.float64)
    params = np.concatenate([params_i, params_d], axis=1)

    # FIXME: take directory list and CONTCAR/XDATCAR info
    structure_list = [os.path.join(os.path.dirname(os.path.realpath(__file__)) + "/test/CONTCAR")]

    for item in structure_list:
        atoms = io.read(item)
        
        cart_p  = _gen_2Darray_for_ffi(atoms.positions, ffi)
        scale_p = _gen_2Darray_for_ffi(atoms.get_scaled_positions(), ffi)
        cell_p  = _gen_2Darray_for_ffi(atoms.cell, ffi)
        
        atom_i = np.array(atoms.get_chemical_symbols())
        for j,jtem in enumerate(set(atom_i)):
            atom_i[atom_i==jtem] = j+1 # FIXME: index starting number 0 or 1?
        atom_i = atom_i.astype(np.int)
        atom_i_p = ffi.cast("int *", atom_i.ctypes.data)

        atom_num = len(atoms.positions)
        param_num = len(params_i)

        print("== check for C extension code ==\n")

        res = dict()
        res['x'] = np.zeros([atom_num, param_num]).astype(np.float64)
        res['dx'] = np.zeros([atom_num, atom_num * param_num * 3]).astype(np.float64)
        res['params'] = params

        x_p = _gen_2Darray_for_ffi(res['x'], ffi, np.float64)
        dx_p = _gen_2Darray_for_ffi(res['dx'], ffi, np.float64) # TODO: change the dimension of res['dx']

        lib.calculate_sf(cell_p, cart_p, scale_p, atom_i_p, params_ip, params_dp, atom_num, param_num, x_p, dx_p)

        #feature_dict = calculate_feature(item)
        print(res['x'][0])
        print(list(res['dx'][1]))



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
