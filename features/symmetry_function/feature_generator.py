from __future__ import print_function
from __future__ import division
import os
from mpi4py import MPI
import numpy as np
import six
from six.moves import cPickle as pickle
from ase import io
from cffi import FFI

# TODO: Different atom can get different symmetry function parameter
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
        for line in fil:
            tmp = line.split()
            params_i += [list(map(int,   tmp[:3]))]
            params_d += [list(map(float, tmp[3:]))]

    params_i = np.asarray(params_i, dtype=np.intc, order='C')
    params_d = np.asarray(params_d, dtype=np.float64, order='C')

    return params_i, params_d

def feature_generator(structure_list, param_list):
    # structure_list = filename which contains the directory info for each structure file
    #   [file format]
    #   directory1/filename [snapshot info (ASE format)]
    # param_list = dictionary of parameter filename list. {atom_type: file_name}
    # FIXME: change the file format
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
    with open(param_list) as fil:
        params_set = dict()
        for line in fil:
            tmp = line.strip('\n').split()
            params_set[tmp[0]] = _read_params(tmp[1])
    atom_list = params_set.keys()

    """
    params_ip = _gen_2Darray_for_ffi(params_i, ffi, "int")
    params_dp = _gen_2Darray_for_ffi(params_d, ffi)
    params = np.concatenate([params_i, params_d], axis=1)
    param_num = len(params_i)
    """

    # read structure info
    with open(structure_list, 'r') as fil:
        structures = list()
        for line in fil:
            structures.append(line.strip('\n').split())

    for item in structures:
        # FIXME: add another input type
        # TODO: modulization. currently, we suppose that the input file is VASP based.
        if len(item) == 1:
            index_str = '-1:'
        else:
            index_str = item[1]
        snapshots = io.read(item[0], index=index_str, force_consistent=True)

        for atoms in snapshots:
            cart = np.copy(atoms.positions, order='C')
            scale = np.copy(atoms.get_scaled_positions(), order='C')
            cell = np.copy(atoms.cell, order='C')

            cart_p  = _gen_2Darray_for_ffi(cart, ffi)
            scale_p = _gen_2Darray_for_ffi(scale, ffi)
            cell_p  = _gen_2Darray_for_ffi(cell, ffi)
        
            atom_num = len(atoms.positions)
            symbols = np.array(atoms.get_chemical_symbols())
            atom_i = np.zeros([len(symbols)], dtype=np.intc, order='C')
            type_num = dict()
            type_idx = dict()
            for j,jtem in enumerate(atom_list):
                tmp = symbols==jtem
                atom_i[tmp] = j+1
                type_num[jtem] = np.sum(tmp)
                # if atom indexs are sorted by atom type,
                # indexs are sorted in this part.
                # if not, it could generate bug in training process for force training
                type_idx[jtem] = np.arange(atom_num)[tmp]
            atom_i_p = ffi.cast("int *", atom_i.ctypes.data)

            res = dict()
            res['x'] = dict()
            res['dx'] = dict()
            res['F'] = dict()
            res['params'] = dict()
            res['N'] = type_num
            res['E'] = atoms.get_total_energy()

            for j,jtem in enumerate(atom_list):
                q = type_num[jtem] // size
                r = type_num[jtem] %  size

                begin = rank * q + min(rank, r)
                end = begin + q
                if r > rank:
                    end += 1

                cal_atoms = np.asarray(type_idx[jtem][begin:end], dtype=np.intc, order='C')
                cal_num = len(cal_atoms)
                cal_atoms_p = ffi.cast("int *", cal_atoms.ctypes.data)

                params_ip = _gen_2Darray_for_ffi(params_set[jtem][0], ffi, "int")
                params_dp = _gen_2Darray_for_ffi(params_set[jtem][1], ffi)
                params = np.concatenate(params_set[jtem], axis=1)
                param_num = len(params)

                x = np.zeros([cal_num, param_num], dtype=np.float64, order='C')
                dx = np.zeros([cal_num, atom_num * param_num * 3], dtype=np.float64, order='C')

                x_p = _gen_2Darray_for_ffi(x, ffi)
                dx_p = _gen_2Darray_for_ffi(dx, ffi)

                lib.calculate_sf(cell_p, cart_p, scale_p, atom_i_p, atom_num, cal_atoms_p, cal_num, params_ip, params_dp, param_num, x_p, dx_p)
                comm.barrier()

                res['x'][jtem] = np.array(comm.gather(x, root=0))
                res['dx'][jtem] = np.array(comm.gather(dx, root=0))

                if rank == 0:
                    res['x'][jtem] = np.concatenate(res['x'][jtem], axis=0).reshape([type_num[jtem], param_num])
                    res['dx'][jtem] = np.concatenate(res['dx'][jtem], axis=0).reshape([type_num[jtem], param_num, atom_num, 3])
                    res['F'][jtem] = atoms.get_forces()[type_idx[jtem]]
                    res['params'][jtem] = params
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

            if rank == 0:
                with open(os.path.join(os.path.dirname(os.path.realpath(__file__)) + "/test/test1.pickle"), "wb") as fil:
                    pickle.dump(res, fil, pickle.HIGHEST_PROTOCOL)  # TODO: directory setting?

if __name__ == "__main__":
    # for test
    feature_generator('as', 'as')
