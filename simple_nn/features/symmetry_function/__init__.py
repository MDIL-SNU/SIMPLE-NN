from __future__ import print_function
from __future__ import division
import os, sys
import tensorflow as tf
import numpy as np
import six
from six.moves import cPickle as pickle
from ase import io
from cffi import FFI
from ...utils import _gen_2Darray_for_ffi, compress_outcar, _generate_scale_file

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

    def _write_tfrecords(self, res, record_name, atomic_weights=False):
        # TODO: after stabilize overall tfrecord related part,
        # this part will replace the part of original 'res' dict
         
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        
        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _gen_1dsparse(arr):
            non_zero = (arr != 0)
            return np.arange(arr.shape[0])[non_zero].astype(np.int32), arr[non_zero], np.array(arr.shape).astype(np.int32)
        
        feature = {
            'E':_bytes_feature(np.array([res['E']]).tobytes()),
            'F':_bytes_feature(res['F'].tobytes())
        }
    
        if atomic_weights:
            feature['atomic_weights'] = _bytes_feature(res['atomic_weights'].tobytes())
        
        for item in self.parent.inputs['atom_types']:
            feature['x_'+item] = _bytes_feature(res['x'][item].tobytes())
            feature['N_'+item] = _int64_feature(res['N'][item])
            feature['params_'+item] = _bytes_feature(res['params'][item].tobytes())

            dx_indices, dx_values, dx_dense_shape = _gen_1dsparse(res['dx'][item].reshape([-1]))
            
            feature['dx_indices_'+item] = _bytes_feature(dx_indices.tobytes())
            feature['dx_values_'+item] = _bytes_feature(dx_values.tobytes())
            feature['dx_dense_shape_'+item] = _bytes_feature(dx_dense_shape.tobytes())

        example = tf.train.Example(
            features=tf.train.Features(
                feature=feature
            )
        )
        
        with tf.python_io.TFRecordWriter(record_name) as writer:
            writer.write(example.SerializeToString())


    def _parse_data(self, serialized, inp_size, valid=False, serialized_aw=None):
        features = {
            'E': tf.FixedLenFeature([], dtype=tf.string),
            'F': tf.FixedLenFeature([], dtype=tf.string),
        }
 
        for item in self.parent.inputs['atom_types']:
            features['x_'+item] = tf.FixedLenFeature([], dtype=tf.string)
            features['N_'+item] = tf.VarLenFeature(tf.int64)
            features['params_'+item] = tf.FixedLenFeature([], dtype=tf.string)
            features['dx_indices_'+item] = tf.FixedLenFeature([], dtype=tf.string)
            features['dx_values_'+item] = tf.FixedLenFeature([], dtype=tf.string)
            features['dx_dense_shape_'+item] = tf.FixedLenFeature([], dtype=tf.string)
 
        read_data = tf.parse_single_example(serialized=serialized, features=features)
 
        res = dict()
 
        res['E'] = tf.decode_raw(read_data['E'], tf.float64)
        res['F'] = tf.reshape(tf.decode_raw(read_data['F'], tf.float64), [-1, 3])
        res['tot_num'] = 0.
        for item in self.parent.inputs['atom_types']:
            res['x_'+item] = tf.reshape(tf.decode_raw(read_data['x_'+item], tf.float64), [-1, inp_size[item]])
            #res['N_'+item] = read_data['N_'+item]
            res['N_'+item] = tf.shape(res['x_'+item])[0]
            res['tot_num'] += tf.cast(tf.shape(res['x_'+item])[0], tf.float64)
 
            res['dx_'+item] = tf.sparse_to_dense(
                sparse_indices=tf.decode_raw(read_data['dx_indices_'+item], tf.int32),
                output_shape=tf.decode_raw(read_data['dx_dense_shape_'+item], tf.int32),
                sparse_values=tf.decode_raw(read_data['dx_values_'+item], tf.float64)
            )
            res['dx_'+item] = tf.reshape(res['dx_'+item], [tf.shape(res['x_'+item])[0], inp_size[item], -1, 3])
            res['partition_'+item] = tf.ones([tf.shape(res['x_'+item])[0]])
 
        # TODO: seg_id, dynamic_partition_id, 
        #res['tot_num'] = tf.sparse_reduce_sum(tf.sparse_concat(0, res['tot_num']))
        res['partition'] = tf.ones([tf.cast(res['tot_num'], tf.int32)])

        if serialized_aw is not None:
            if valid:
                res['atomic_weights'] = tf.ones([tf.cast(res['tot_num'], tf.float64)])
            else:
                features_aw = {'atomic_weights':tf.FizedLenFeature([], dtype=tf.string)}
                read_data_aw = tf.parse_single_example(serialized=serialized_aw, features=features_aw)
                res['atomic_weights'] = tf.decode_raw(read_data_aw['atomic_weights'], tf.float64)
 
        return res


    def _tfrecord_input_fn(self, filename_queue, inp_size, batch_size=1, valid=False, atomic_weights=False):
        dataset = tf.data.TFRecordDataset(filename_queue)

        if (valid == False) and (atomic_weights == True):
            aw_filename_queue = [item.replace('.tfrecord', '_atomic_weights.tfrecord') for item in filename_queue]
            dataset = tf.data.Dataset.zip((dataset, tf.data.TFRecordDataset(aw_filename_queue)))
            dataset = dataset.map(lambda x, y: self._parse_data(x, inp_size, valid=False, serialized_aw=y))
        else:
            dataset = dataset.map(lambda x: self._parse_data(x, inp_size, valid=False, serialized_aw=None))

        batch_dict = dict()
        batch_dict['E'] = [None]
        batch_dict['F'] = [None, 3]
        batch_dict['tot_num'] = []
        batch_dict['partition'] = [None]

        if atomic_weights:
            batch_dict['atomic_weights'] = [None]

        for item in self.parent.inputs['atom_types']:
            batch_dict['x_'+item] = [None, inp_size[item]]
            batch_dict['N_'+item] = []
            batch_dict['dx_'+item] = [None, inp_size[item], None, 3]
            batch_dict['partition_'+item] = [None]
 
        if valid:
            dataset = dataset.padded_batch(1, batch_dict)
            iterator = dataset.make_initializable_iterator()
        else:
            dataset = dataset.shuffle(buffer_size=200)
            # order of repeat and batch?
            dataset = dataset.repeat()
            dataset = dataset.padded_batch(batch_size, batch_dict)
            iterator = dataset.make_initializable_iterator()
            
        return iterator  


    def preprocess(self, filename_queue, inp_size, calc_scale=True, get_atomic_weights=None, **kwargs):
        
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        aw_tag = False
        train_iter = self._tfrecord_input_fn(filename_queue, inp_size, valid=True, atomic_weights=False)
        next_elem = train_iter.get_next()        

        feature_list = dict()
        idx_list = dict()
        for item in self.parent.inputs['atom_types']:
            feature_list[item] = list()
            idx_list[item] = list()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(train_iter.initializer)
            while True:
                try:
                    tmp_features = sess.run(next_elem)
                    for item in self.parent.inputs['atom_types']:
                        feature_list[item].append(tmp_features['x_'+item][0])
                        idx_list[item].append(np.ones(tmp_features['N_'+item][0]))
                except tf.errors.OutOfRangeError:
                    break

        for item in self.parent.inputs['atom_types']:
            feature_list[item] = np.concatenate(feature_list[item], axis=0)
            idx_list[item] = np.concatenate(idx_list[item])
    
        scale = None
        if calc_scale:
            scale = _generate_scale_file(feature_list, self.parent.inputs['atom_types'], inp_size)
        else:
            scale = pickle_load('./scale_factor')
        
        if callable(get_atomic_weights):
            aw_tag = True
            atomic_weights = get_atomic_weights(feature_list, scale, atom_types, idx_list, **kwarg)
            for i,item in enumerate(filename_queue):
                tmp_name = item.replace('.tfrecord', '_atomic_weights.tfrecord')

                aw_idx = (atomic_weights[1,:] == i)
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={'atomic_weights': _byte_feature(atomic_weights[0, aw_idx].tobytes())}
                    )
                )                

                with tf.python_io.TFRecordWriter(tmp_name) as writer:
                    writer.write(example.SerializeToString())
                   
        return scale, aw_tag
 

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
                    #_write_tfrecords(res, self.parent.inputs['atom_types'], sess, tmp_filename)
                    self._write_tfrecords(res, tmp_filename)
                    #with open(tmp_filename, "wb") as fil:
                    #    pickle.dump(res, fil, pickle.HIGHEST_PROTOCOL)  

                    train_dir.write('{}\n'.format(tmp_filename))
                    tmp_endfile = tmp_filename
                    data_idx += 1

            if comm.rank == 0:
                self.parent.logfile.write(': ~{}\n'.format(tmp_endfile))

        if comm.rank == 0:
            train_dir.close()
