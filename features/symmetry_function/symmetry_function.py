from mpi4py import MPI
from six.moves import cPickle as pickle
from ase import io
from cffi import FFI

def _gen_2Darray_for_ffi(arr):
    shape = arr.shape
    arr_p = ffi.new("double *[%d]" % shape[0])
    for i in range(shape[0]):
        arr_p[i] = ffi.cast("double *", arr[i].ctypes.data)
    return arr_p

def feature_generator(structure_list, param_list):
    # get directory info for each structures and parameter list of symmetry functions
    
    ffi = FFI()

    with open(param_list, 'r') as fil:
        params = None

    params_p = _gen_2Darray_for_ffi(params)

    # TODO: library read using ffi

    for item in structure_list:
        
        atoms = io.read()
        posi_p = _gen_2Darray_for_ffi(atoms.positions)
        cell_p = _gen_2Darray_for_ffi(atoms.cell)
        atom_num = len(atoms.positions)
        param_num = len(params)

        res = dict()
        res['x'] = np.zeros([atom_num, param_num]).astype(np.float64)
        res['dx'] = np.zeros([atom_num, atom_num, param_num, 3]).astype(np.float64)
        res['params'] = params

        x_p = _gen_2Darray_for_ffi(res['x'])
        dx_p = _gen_2Darray_for_ffi(res['dx']) # TODO: change the dimension of res['dx']

        feature_dict = calculate_feature(item)




        pickle.dump(feature_dict, _name_, pickle.HIGHST_PROTOCOL)  # TODO: directory setting?
    return 0


def calculate_feature(structure):
    comm = MPI.COMM_WORLD
    atoms = io.read(structure)  # TODO: add structure file information

    feature_dict = dict()
    feature_dict['sf'], feature_dict['dsf'] = _c_extension_for_calculating_sf_(atoms, comm)
    # TODO: make c extension

    return feature_dict
