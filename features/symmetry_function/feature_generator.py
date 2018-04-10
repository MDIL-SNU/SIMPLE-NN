from __future__ import print_function
from __future__ import division
import os
from mpi4py import MPI
import numpy as np
import six
from six.moves import cPickle as pickle
from ase import io
from cffi import FFI

def _gen_2Darray_for_ffi(arr, ffi, cdata="double"):
    # Function to generate 2D pointer for cffi  
    shape = arr.shape
    arr_p = ffi.new(cdata + " *[%d]" % shape[0])
    for i in range(shape[0]):
        arr_p[i] = ffi.cast(cdata + " *", arr[i].ctypes.data)
    return arr_p

def _read_params(filename):
    params_i = list()
    params_d = list()
    with open(filename, 'r') as fil:
        atom_list = fil.readline().strip('\n').split()
        for line in fil:
            tmp = line.split()
            params_i += [list(map(int,   tmp[:3]))]
            params_d += [list(map(float, tmp[3:]))]

    return params_i, params_d, atom_list

def feature_generator(structure_list, param_list):
    # get directory info for each structures and parameter list of symmetry functions

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
       
    ffi = FFI()
    ffi.cdef("""void calculate_sf(double **, double **, double **,
                                  int *, int, int*, int,
                                  int**, double **, int, 
                                  double**, double**);""")
    lib = ffi.dlopen(os.path.join(os.path.dirname(os.path.realpath(__file__)) + "/libsymf.so"))

    # parameter list
    # [symmetry function types, atom type 1, atom type 2, cutoff, parameter 1, 2, ...]
    #params_i, params_d, atom_list = _read_params(os.path.join(os.path.dirname(os.path.realpath(__file__)) + "/test/inp_fsymf_Ni2Si"))
    params_i, params_d, atom_list = _read_params(param_list)
    params_i = np.asarray(params_i, dtype=np.int32, order='C')
    params_d = np.asarray(params_d, dtype=np.float64, order='C')

    params_ip = _gen_2Darray_for_ffi(params_i, ffi, "int")
    params_dp = _gen_2Darray_for_ffi(params_d, ffi)
    params = np.concatenate([params_i, params_d], axis=1)
    param_num = len(params_i)

    # FIXME: take directory list and CONTCAR/XDATCAR info
    #structure_list = [os.path.join(os.path.dirname(os.path.realpath(__file__)) + "/test/CONTCAR_Ni2Si")]
    with open(structure_list, 'r') as fil:
        structures = list()
        for line in fil:
            structures.append(line.strip('\n').split())

    for item in structures:
        # FIXME: add another input type
        if item[1] == 'CONTCAR':
            snapshots = [io.read(item[0])]
        elif item[1] == 'XDATCAR':
            index_str = '{}:{}:{}'.format(item[2], item[3], item[4])
            snapshots = io.read(item[0], index=index_str)
        
        for atoms in snapshots:
            cart = np.copy(atoms.positions, order='C')
            scale = np.copy(atoms.get_scaled_positions(), order='C')
            cell = np.copy(atoms.cell, order='C')

            cart_p  = _gen_2Darray_for_ffi(cart, ffi)
            scale_p = _gen_2Darray_for_ffi(scale, ffi)
            cell_p  = _gen_2Darray_for_ffi(cell, ffi)
        
            symbols = np.array(atoms.get_chemical_symbols())
            atom_i = np.zeros([len(symbols)], dtype=np.int32, order='C')
            for j,jtem in enumerate(atom_list):
                atom_i[symbols==jtem] = j+1
            atom_i_p = ffi.cast("int *", atom_i.ctypes.data)

            atom_num = len(atoms.positions)

            q = atom_num // size
            r = atom_num %  size

            begin = rank * q + min(rank, r)
            end = begin + q
            if r > rank:
                end += 1

            cal_atoms = np.asarray(range(begin, end), dtype=np.int32, order='C')
            cal_num = len(cal_atoms)
            cal_atoms_p = ffi.cast("int *", cal_atoms.ctypes.data)

            x = np.zeros([cal_num, param_num], dtype=np.float64, order='C')
            dx = np.zeros([cal_num, atom_num * param_num * 3], dtype=np.float64, order='C')

            x_p = _gen_2Darray_for_ffi(x, ffi)
            dx_p = _gen_2Darray_for_ffi(dx, ffi)

            lib.calculate_sf(cell_p, cart_p, scale_p, atom_i_p, atom_num, cal_atoms_p, cal_num, params_ip, params_dp, param_num, x_p, dx_p)
            comm.barrier()

            res = dict()
            res['x'] = np.array(comm.gather(x, root=0))
            res['dx'] = np.array(comm.gather(dx, root=0))
            res['params'] = params

            if rank == 0:
                res['x'] = res['x'].reshape([atom_num, param_num])
                res['dx'] = res['dx'].reshape([atom_num, param_num, atom_num, 3])
                # FIXME: change the data structure

                # FIXME: for test
                """
                with open(os.path.join(os.path.dirname(os.path.realpath(__file__)) + "/test/structure1.pickle"), "rb") as fil:
                    if six.PY2:
                        refdat = pickle.load(fil)
                    elif six.PY3:
                        refdat = pickle.load(fil, encoding='latin1')

                dx_cal = res['dx'][:4]
                dx_ref = refdat['dsym']['Si']
                print(np.sum((dx_cal - dx_ref) < 1e-13))
                """

                with open(os.path.join(os.path.dirname(os.path.realpath(__file__)) + "/test/test1.pickle"), "wb") as fil:
                    pickle.dump(res, fil, pickle.HIGHEST_PROTOCOL)  # TODO: directory setting?

if __name__ == "__main__":
    # for test
    feature_generator('as', 'as')
