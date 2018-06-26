from __future__ import print_function
from __future__ import division
import os, sys
import tensorflow as tf
import numpy as np
import six
from six.moves import cPickle as pickle
from ase import io
from cffi import FFI
from ...utils import _gen_2Darray_for_ffi, compress_outcar

class DummyMPI(object):
    rank = 0
    size = 1

    def barrier(self):
        pass

    def gather(self, data, root=0):
        return data

class MPI4PY(object):
    def __init__(self):
        from mpi4py import MPI
        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

    def barrier(self):
        self.comm.barrier()

    def gather(self, data, root=0):
        return self.comm.gather(data, root=0)

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

class Symmetry_function(object):
    def __init__(self, inputs=None):
        self.parent = None
        self.key = 'symmetry_function'
        self.default_inputs = {'symmetry_function': 
                                  {
                                      'params': dict(),
                                      'compress_outcar':True
                                  }
                              }
        self.structure_list = './str_list'
        self.train_data_list = './train_list'

    def _write_tfrecords(self, res, sess, record_name):
        # TODO: after stabilize overall tfrecord related part,
        # this part will replace the part of original 'res' dict
         
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        
        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        feature = {
            'E':_bytes_feature(np.array([res['E']]).tobytes()),
            'F':_bytes_feature(res['F'].tobytes())
        }

        for item in self.parent.inputs['atom_types']:
            feature['x_'+item] = _bytes_feature(res['x'][item].tobytes())
            feature['N_'+item] = _int64_feature(res['N'][item])
            feature['params_'+item] = _bytes_feature(res['params'][item].tobytes())

            dx_sparse = tf.contrib.layers.dense_to_sparse(res['dx'][item].reshape([-1]))
            
            feature['dx_indices_'+item] = _bytes_feature(sess.run(dx_sparse.indices).astype(np.uint32).tobytes())
            feature['dx_values_'+item] = _bytes_feature(sess.run(dx_sparse.values).tobytes())
            feature['dx_dense_shape_'+item] = _bytes_feature(sess.run(dx_sparse.dense_shape).tobytes())

            del dx_sparse

        example = tf.train.Example(
            features=tf.train.Features(
                feature=feature
            )
        )

        with tf.python_io.TFRecordWriter(record_name) as writer:
            writer.write(example.SerializeToString())


    def generate(self):
        self.inputs = self.parent.inputs['symmetry_function']

        if 'mpi4py' in sys.modules:
            comm = MPI4PY()
        else:
            comm = DummyMPI()

        ffi = FFI()
        ffi.cdef("""void calculate_sf(double **, double **, double **,
                                      int *, int, int*, int,
                                      int**, double **, int, 
                                      double**, double**);""")
        lib = ffi.dlopen(os.path.join(os.path.dirname(os.path.realpath(__file__)) + "/libsymf.so"))

        if comm.rank == 0:
            train_dir = open(self.train_data_list, 'w')
            sess = tf.Session()

        # Get structure list to calculate  
        structures = list()
        with open(self.structure_list, 'r') as fil:
            for line in fil:
                structures.append(line.strip().split())

        # Get parameter list for each atom types
        params_set = dict()
        for item in self.parent.inputs['atom_types']:
            params_set[item] = dict()
            params_set[item]['i'], params_set[item]['d'] = \
                _read_params(self.inputs['params'][item])
            params_set[item]['ip'] = _gen_2Darray_for_ffi(params_set[item]['i'], ffi, "int")
            params_set[item]['dp'] = _gen_2Darray_for_ffi(params_set[item]['d'], ffi)
            params_set[item]['total'] = np.concatenate((params_set[item]['i'], params_set[item]['d']), axis=1)
            params_set[item]['num'] = len(params_set[item]['total'])
            
        data_idx = 1
        for item in structures:
            # FIXME: add another input type
            
            if len(item) == 1:
                index = 0
                if comm.rank == 0:
                    self.parent.logfile.write('{} 0'.format(item[0]))
            else:
                if ':' in item[1]:
                    index = item[1]
                else:
                    index = int(item[1])

                if comm.rank == 0:
                    self.parent.logfile.write('{} {}'.format(item[0], item[1]))

            if self.inputs['compress_outcar']:
                tmp_name = compress_outcar(item[0])
                snapshots = io.read(tmp_name, index=index, force_consistent=True)
                os.remove(tmp_name)
            else:    
                snapshots = io.read(item[0], index=index, force_consistent=True) 

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
                for j,jtem in enumerate(self.parent.inputs['atom_types']):
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
                res['params'] = dict()
                res['N'] = type_num
                res['E'] = atoms.get_total_energy()
                res['F'] = atoms.get_forces()

                for j,jtem in enumerate(self.parent.inputs['atom_types']):
                    q = type_num[jtem] // comm.size
                    r = type_num[jtem] %  comm.size

                    begin = comm.rank * q + min(comm.rank, r)
                    end = begin + q
                    if r > comm.rank:
                        end += 1

                    cal_atoms = np.asarray(type_idx[jtem][begin:end], dtype=np.intc, order='C')
                    cal_num = len(cal_atoms)
                    cal_atoms_p = ffi.cast("int *", cal_atoms.ctypes.data)

                    x = np.zeros([cal_num, params_set[jtem]['num']], dtype=np.float64, order='C')
                    dx = np.zeros([cal_num, atom_num * params_set[jtem]['num'] * 3], dtype=np.float64, order='C')

                    x_p = _gen_2Darray_for_ffi(x, ffi)
                    dx_p = _gen_2Darray_for_ffi(dx, ffi)

                    lib.calculate_sf(cell_p, cart_p, scale_p, \
                                     atom_i_p, atom_num, cal_atoms_p, cal_num, \
                                     params_set[jtem]['ip'], params_set[jtem]['dp'], params_set[jtem]['num'], \
                                     x_p, dx_p)
                    comm.barrier()


                    if comm.rank == 0:
                        if type_num[jtem] != 0:
                            res['x'][jtem] = np.array(comm.gather(x, root=0))
                            res['dx'][jtem] = np.array(comm.gather(dx, root=0))
                            res['x'][jtem] = np.concatenate(res['x'][jtem], axis=0).reshape([type_num[jtem], params_set[jtem]['num']])
                            res['dx'][jtem] = np.concatenate(res['dx'][jtem], axis=0).\
                                                reshape([type_num[jtem], params_set[jtem]['num'], atom_num, 3])
                            res['params'][jtem] = params_set[jtem]['total']

                if comm.rank == 0:
                    data_dir = "./data/"
                    if not os.path.exists(data_dir):
                        os.makedirs(data_dir)
                    #tmp_filename = os.path.join(data_dir, "data{}.pickle".format(data_idx))
                    tmp_filename = os.path.join(data_dir, "data{}.tfrecord".format(data_idx))

                    # TODO: add tfrecord writing part
                    self._write_tfrecords(res, sess, tmp_filename)
                    #with open(tmp_filename, "wb") as fil:
                    #    pickle.dump(res, fil, pickle.HIGHEST_PROTOCOL)  

                    train_dir.write('{}\n'.format(tmp_filename))
                    tmp_endfile = tmp_filename
                    data_idx += 1

            if comm.rank == 0:
                self.parent.logfile.write(': ~{}\n'.format(tmp_endfile))

        if comm.rank == 0:
            train_dir.close()
            sess.close()
