from __future__ import print_function
from __future__ import division
import os, sys
import tensorflow as tf
import numpy as np
import six
from six.moves import cPickle as pickle
from ase import io
from cffi import FFI
from ...utils import _gen_2Darray_for_ffi, compress_outcar, _generate_scale_file, \
                     _make_full_featurelist, _make_data_list, _make_str_data_list, pickle_load
from ...utils import graph as grp
from braceexpand import braceexpand

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
                                      'compress_outcar':True,
                                      'data_per_tfrecord': 150,
                                      'valid_rate': 0.1,
                                      'remain_pickle': False,
                                      'continue': False,
                                      'add_atom_idx': True, # For backward compatability
                                      'num_parallel_calls': 5,
                                      'atomic_weights': {
                                          'type': None,
                                          'params': dict(),
                                      },
                                      'weight_modifier': {
                                          'type': None,
                                          'params': dict(),
                                      },
                                  }
                              }
        self.structure_list = './str_list'
        self.pickle_list = './pickle_list'
        self.train_data_list = './train_list'
        self.valid_data_list = './valid_list'

    def set_inputs(self):
        self.inputs = self.parent.inputs['symmetry_function']

    def _write_tfrecords(self, res, writer, use_force=False, atomic_weights=False):
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
            'tot_num':_bytes_feature(res['tot_num'].astype(np.float64).tobytes()),
            'partition':_bytes_feature(res['partition'].tobytes()),
            'struct_type':_bytes_feature(six.b(res['struct_type'])),
            'struct_weight': _bytes_feature(np.array([res['struct_weight']]).tobytes())
        }
    
        if use_force:
            feature['F'] = _bytes_feature(res['F'].tobytes())
            if self.inputs['add_atom_idx']:
                feature['atom_idx'] = _bytes_feature(res['atom_idx'].tobytes())
#            if atomic_weights:
#                feature['atomic_weights'] = _bytes_feature(res['atomic_weights'].tobytes())
        
        for item in self.parent.inputs['atom_types']:
            feature['x_'+item] = _bytes_feature(res['x'][item].tobytes())
            feature['N_'+item] = _bytes_feature(res['N'][item].tobytes())
            feature['params_'+item] = _bytes_feature(res['params'][item].tobytes())

            if use_force:
                dx_indices, dx_values, dx_dense_shape = _gen_1dsparse(res['dx'][item].reshape([-1]))
            
                feature['dx_indices_'+item] = _bytes_feature(dx_indices.tobytes())
                feature['dx_values_'+item] = _bytes_feature(dx_values.tobytes())
                feature['dx_dense_shape_'+item] = _bytes_feature(dx_dense_shape.tobytes())

            feature['partition_'+item] = _bytes_feature(res['partition_'+item].tobytes())

            if atomic_weights:
                feature['atomic_weights_'+item] = _bytes_feature(res['atomic_weights'][item].tobytes())

        example = tf.train.Example(
            features=tf.train.Features(
                feature=feature
            )
        )
        
        writer.write(example.SerializeToString())


    def _parse_data(self, serialized, inp_size, use_force=False, atomic_weights=False):
        features = {
            'E': tf.FixedLenFeature([], dtype=tf.string),
            'tot_num': tf.FixedLenFeature([], dtype=tf.string),
            'partition': tf.FixedLenFeature([], dtype=tf.string),
            'struct_type': tf.FixedLenSequenceFeature([], dtype=tf.string, allow_missing=True),
            'struct_weight': tf.FixedLenFeature([], dtype=tf.string, default_value=np.array([1.0]).tobytes()),
        }
 
        for item in self.parent.inputs['atom_types']:
            features['x_'+item] = tf.FixedLenFeature([], dtype=tf.string)
            features['N_'+item] = tf.FixedLenFeature([], dtype=tf.string)
            features['params_'+item] = tf.FixedLenFeature([], dtype=tf.string)
            if use_force:
                features['dx_indices_'+item] = tf.FixedLenFeature([], dtype=tf.string)
                features['dx_values_'+item] = tf.FixedLenFeature([], dtype=tf.string)
                features['dx_dense_shape_'+item] = tf.FixedLenFeature([], dtype=tf.string)
            features['partition_'+item] = tf.FixedLenFeature([], dtype=tf.string)
            if atomic_weights:
                features['atomic_weights_'+item] = tf.FixedLenFeature([], dtype=tf.string)

        if use_force:
            features['F'] = tf.FixedLenFeature([], dtype=tf.string)
            if self.inputs['add_atom_idx']:
                features['atom_idx'] = tf.FixedLenFeature([], dtype=tf.string)
            #if atomic_weights:
            #    features['atomic_weights'] = tf.FixedLenFeature([], dtype=tf.string)

        read_data = tf.parse_single_example(serialized=serialized, features=features)
        #read_data = tf.parse_example(serialized=serialized, features=features)
 
        res = dict()
 
        res['E'] = tf.decode_raw(read_data['E'], tf.float64)
        res['tot_num'] = tf.decode_raw(read_data['tot_num'], tf.float64)
        res['partition'] = tf.decode_raw(read_data['partition'], tf.int32)
        res['struct_type'] = read_data['struct_type']
        res['struct_weight'] = tf.decode_raw(read_data['struct_weight'], tf.float64)
        # For backward compatibility...
        res['struct_type'] = tf.cond(tf.equal(tf.shape(res['struct_type'])[0], 0),
                                     lambda: tf.constant(['None']),
                                     lambda: res['struct_type'])

        for item in self.parent.inputs['atom_types']:
            res['N_'+item] = tf.decode_raw(read_data['N_'+item], tf.int64)

            res['x_'+item] = tf.cond(tf.equal(res['N_'+item][0], 0),
                                     lambda: tf.zeros([0, inp_size[item]], dtype=tf.float64),
                                     lambda: tf.reshape(tf.decode_raw(read_data['x_'+item], tf.float64), [-1, inp_size[item]]))

            if use_force:
                res['dx_'+item] = tf.cond(tf.equal(res['N_'+item][0], 0),
                                          lambda: tf.zeros([0, inp_size[item], 0, 3], dtype=tf.float64),
                                          lambda: tf.reshape(
                                            tf.sparse_to_dense(
                                                sparse_indices=tf.decode_raw(read_data['dx_indices_'+item], tf.int32),
                                                output_shape=tf.decode_raw(read_data['dx_dense_shape_'+item], tf.int32),
                                                sparse_values=tf.decode_raw(read_data['dx_values_'+item], tf.float64)), 
                                            [tf.shape(res['x_'+item])[0], inp_size[item], -1, 3]))
 
            res['partition_'+item] = tf.decode_raw(read_data['partition_'+item], tf.int32)

            if atomic_weights:
                res['atomic_weights_'+item] = tf.decode_raw(read_data['atomic_weights_'+item], tf.float64)

        if use_force: 
            res['F'] = tf.reshape(tf.decode_raw(read_data['F'], tf.float64), [-1, 3])
            if self.inputs['add_atom_idx']:
                res['atom_idx'] = tf.reshape(tf.decode_raw(read_data['atom_idx'], tf.int32), [-1, 1])
            #if atomic_weights:
            #    res['atomic_weights'] = tf.decode_raw(read_data['atomic_weights'], tf.float64)
 
        return res


    def _tfrecord_input_fn(self, filename_queue, inp_size, batch_size=1, use_force=False, valid=False, cache=False, full_batch=False, atomic_weights=False):
        dataset = tf.data.TFRecordDataset(filename_queue)
        #dataset = dataset.cache() # for test
        dataset = dataset.map(lambda x: self._parse_data(x, inp_size, use_force=use_force, atomic_weights=atomic_weights), 
                              num_parallel_calls=self.inputs['num_parallel_calls'])
        if cache:
            dataset = dataset.take(-1).cache() # for test

        batch_dict = dict()
        batch_dict['E'] = [None]
        batch_dict['tot_num'] = [None]
        batch_dict['partition'] = [None]
        batch_dict['struct_type'] = [None]
        batch_dict['struct_weight'] = [None]

        if use_force:
            batch_dict['F'] = [None, 3]
            if self.inputs['add_atom_idx']:
                batch_dict['atom_idx'] = [None, 1]
            #if atomic_weights:
            #    batch_dict['atomic_weights'] = [None]

        for item in self.parent.inputs['atom_types']:
            batch_dict['x_'+item] = [None, inp_size[item]]
            batch_dict['N_'+item] = [None]
            if use_force:
                batch_dict['dx_'+item] = [None, inp_size[item], None, 3]
            batch_dict['partition_'+item] = [None]
            if atomic_weights:
                batch_dict['atomic_weights_'+item] = [None]
 
        if valid or full_batch:
            dataset = dataset.padded_batch(batch_size, batch_dict)
            dataset = dataset.prefetch(buffer_size=1)
            #dataset = dataset.cache()
            iterator = dataset.make_initializable_iterator()
        else:
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(200, None))
            dataset = dataset.padded_batch(batch_size, batch_dict)
            # prefetch test
            dataset = dataset.prefetch(buffer_size=1)
            iterator = dataset.make_initializable_iterator()
            
        return iterator  


    def preprocess(self, calc_scale=True, use_force=False, get_atomic_weights=None, **kwargs):
        
        # pickle list -> train / valid
        tmp_pickle_train = './pickle_train_list'
        tmp_pickle_valid = './pickle_valid_list'

        if not self.inputs['continue']:
            tmp_pickle_train_open = open(tmp_pickle_train, 'w')
            tmp_pickle_valid_open = open(tmp_pickle_valid, 'w')
            for file_list in _make_str_data_list(self.pickle_list):
                np.random.shuffle(file_list)
                num_pickle = len(file_list)
                num_valid = int(num_pickle * self.inputs['valid_rate'])

                for i,item in enumerate(file_list):
                    if i < num_valid:
                        tmp_pickle_valid_open.write(item + '\n')
                    else:
                        tmp_pickle_train_open.write(item + '\n')
        
            tmp_pickle_train_open.close()
            tmp_pickle_valid_open.close()

        # generate full symmetry function vector
        feature_list_train, idx_list_train = \
            _make_full_featurelist(tmp_pickle_train, self.parent.inputs['atom_types'], 'x')
        feature_list_valid, idx_list_valid = \
            _make_full_featurelist(tmp_pickle_valid, self.parent.inputs['atom_types'], 'x')

        # calculate scale
        scale = None
        if calc_scale:
            scale = _generate_scale_file(feature_list_train, self.parent.inputs['atom_types'])
        else:
            scale = pickle_load('./scale_factor')

        # calculate gdf
        atomic_weights_train = atomic_weights_valid = None
        if callable(get_atomic_weights):
            atomic_weights_train = get_atomic_weights(feature_list_train, feature_list_train, scale, self.parent.inputs['atom_types'], idx_list_train, 
                                                    filename='atomic_weights', **kwargs)
            atomic_weights_valid = get_atomic_weights(feature_list_train, feature_list_valid, scale, self.parent.inputs['atom_types'], idx_list_valid, **kwargs)
        elif isinstance(get_atomic_weights, six.string_types):
            atomic_weights_train = pickle_load(get_atomic_weights)
            atomic_weights_valid = 'ones'

        if atomic_weights_train is None:
            aw_tag = False
        else:
            aw_tag = True
            grp.plot_gdfinv_density(atomic_weights_train, self.parent.inputs['atom_types'])
        
        # train
        tmp_pickle_train_list = _make_data_list(tmp_pickle_train)
        #np.random.shuffle(tmp_pickle_train_list)
        num_tmp_pickle_train = len(tmp_pickle_train_list)
        num_tfrecord_train = int(num_tmp_pickle_train / self.inputs['data_per_tfrecord'])
        train_list = open(self.train_data_list, 'w')

        random_idx = np.arange(num_tmp_pickle_train)        
        #if not self.inputs['continue']:
        np.random.shuffle(random_idx)

        for i,item in enumerate(random_idx):
            ptem = tmp_pickle_train_list[item]
            if i == 0:
                record_name = './data/training_data_{:0>4}_to_{:0>4}.tfrecord'.format(int(i/self.inputs['data_per_tfrecord']), num_tfrecord_train)
                writer = tf.python_io.TFRecordWriter(record_name)
            elif (i % self.inputs['data_per_tfrecord']) == 0:
                writer.close()
                self.parent.logfile.write('{} file saved in {}\n'.format(self.inputs['data_per_tfrecord'], record_name))
                train_list.write(record_name + '\n')
                record_name = './data/training_data_{:0>4}_to_{:0>4}.tfrecord'.format(int(i/self.inputs['data_per_tfrecord']), num_tfrecord_train)
                writer = tf.python_io.TFRecordWriter(record_name)

            tmp_res = pickle_load(ptem)
            if atomic_weights_train is not None:
                tmp_aw = dict()
                for jtem in self.parent.inputs['atom_types']:
                    tmp_idx = (atomic_weights_train[jtem][:,1] == item)
                    tmp_aw[jtem] = atomic_weights_train[jtem][tmp_idx,0]
                #tmp_aw = np.concatenate(tmp_aw)
                tmp_res['atomic_weights'] = tmp_aw
            
            self._write_tfrecords(tmp_res, writer, use_force=use_force, atomic_weights=aw_tag)

            if not self.inputs['remain_pickle']:
                os.remove(ptem)

        writer.close()
        self.parent.logfile.write('{} file saved in {}\n'.format((i%self.inputs['data_per_tfrecord'])+1, record_name))
        train_list.write(record_name + '\n')
        train_list.close()

        if self.inputs['valid_rate'] != 0.0:
            # valid
            tmp_pickle_valid_list = _make_data_list(tmp_pickle_valid)
            #np.random.shuffle(tmp_pickle_valid_list)
            num_tmp_pickle_valid = len(tmp_pickle_valid_list)
            num_tfrecord_valid = int(num_tmp_pickle_valid / self.inputs['data_per_tfrecord'])
            valid_list = open(self.valid_data_list, 'w')

            random_idx = np.arange(num_tmp_pickle_valid)        
            #if not self.inputs['continue']:
            np.random.shuffle(random_idx)

            for i,item in enumerate(random_idx):
                ptem = tmp_pickle_valid_list[item]
                if i == 0:
                    record_name = './data/valid_data_{:0>4}_to_{:0>4}.tfrecord'.format(int(i/self.inputs['data_per_tfrecord']), num_tfrecord_valid)
                    writer = tf.python_io.TFRecordWriter(record_name)
                elif (i % self.inputs['data_per_tfrecord']) == 0:
                    writer.close()
                    self.parent.logfile.write('{} file saved in {}\n'.format(self.inputs['data_per_tfrecord'], record_name))
                    valid_list.write(record_name + '\n')
                    record_name = './data/valid_data_{:0>4}_to_{:0>4}.tfrecord'.format(int(i/self.inputs['data_per_tfrecord']), num_tfrecord_valid)
                    writer = tf.python_io.TFRecordWriter(record_name)

                tmp_res = pickle_load(ptem)
                if atomic_weights_valid == 'ones':
                    tmp_res['atomic_weights'] = np.ones([tmp_res['tot_num']]).astype(np.float64)
                elif atomic_weights_valid is not None:
                    tmp_aw = dict()
                    for jtem in self.parent.inputs['atom_types']:
                        tmp_idx = (atomic_weights_valid[jtem][:,1] == item)
                        tmp_aw[jtem] = atomic_weights_valid[jtem][tmp_idx,0]
                    #tmp_aw = np.concatenate(tmp_aw)
                    tmp_res['atomic_weights'] = tmp_aw

                self._write_tfrecords(tmp_res, writer, use_force=use_force, atomic_weights=aw_tag)

                if not self.inputs['remain_pickle']:
                    os.remove(item)

            writer.close()
            self.parent.logfile.write('{} file saved in {}\n'.format((i%self.inputs['data_per_tfrecord'])+1, record_name))
            valid_list.write(record_name + '\n')
            valid_list.close()
 

    def generate(self):

        if 'mpi4py' in sys.modules:
            comm = MPI4PY()
        else:
            comm = DummyMPI()

        ffi = FFI()
        ffi.cdef("""int calculate_sf(double **, double **, double **,
                                     int *, int, int*, int,
                                     int**, double **, int, 
                                     double**, double**);""")
        lib = ffi.dlopen(os.path.join(os.path.dirname(os.path.realpath(__file__)) + "/libsymf.so"))

        if comm.rank == 0:
            train_dir = open(self.pickle_list, 'w')

        # Get structure list to calculate  
        structures, structure_ind, structure_names, structure_weights = self._parse_strlist()

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
        for item, ind in zip(structures, structure_ind):
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
                cart = np.copy(atoms.get_positions(wrap=True), order='C')
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
                    type_num[jtem] = np.sum(tmp).astype(np.int64)
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
                res['tot_num'] = np.sum(type_num.values())
                res['partition'] = np.ones([res['tot_num']]).astype(np.int32)
                res['E'] = atoms.get_total_energy()
                res['F'] = atoms.get_forces()
                res['struct_type'] = structure_names[ind]
                res['struct_weight'] = structure_weights[ind]
                res['atom_idx'] = atom_i

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

                    errno = lib.calculate_sf(cell_p, cart_p, scale_p, \
                                     atom_i_p, atom_num, cal_atoms_p, cal_num, \
                                     params_set[jtem]['ip'], params_set[jtem]['dp'], params_set[jtem]['num'], \
                                     x_p, dx_p)
                    comm.barrier()
                    if errno == 1:
                        err = "Not implemented symmetry function type."
                        self.parent.logfile.write("Error: {:}\n".format(err))
                        raise NotImplementedError(err)
                    #elif errno == 2:
                    #    err = "Zeta in G4/G5 must be integer."
                    #    self.parent.logfile.write("Error: {:}\n".format(err))
                    #    raise ValueError(err)
                    else:
                        assert errno == 0


                    if comm.rank == 0:
                        if type_num[jtem] != 0:
                            res['x'][jtem] = np.array(comm.gather(x, root=0))
                            res['dx'][jtem] = np.array(comm.gather(dx, root=0))
                            res['x'][jtem] = np.concatenate(res['x'][jtem], axis=0).reshape([type_num[jtem], params_set[jtem]['num']])
                            res['dx'][jtem] = np.concatenate(res['dx'][jtem], axis=0).\
                                                reshape([type_num[jtem], params_set[jtem]['num'], atom_num, 3])
                            res['partition_'+jtem] = np.ones([type_num[jtem]]).astype(np.int32)
                        else:
                            res['x'][jtem] = np.zeros([0, params_set[jtem]['num']])
                            res['dx'][jtem] = np.zeros([0, params_set[jtem]['num'], atom_num, 3])
                            res['partition_'+jtem] = np.ones([0]).astype(np.int32)
                        res['params'][jtem] = params_set[jtem]['total']

                if comm.rank == 0:
                    data_dir = "./data/"
                    if not os.path.exists(data_dir):
                        os.makedirs(data_dir)
                    tmp_filename = os.path.join(data_dir, "data{}.pickle".format(data_idx))
                    #tmp_filename = os.path.join(data_dir, "data{}.tfrecord".format(data_idx))

                    # TODO: add tfrecord writing part
                    #self._write_tfrecords(res, tmp_filename)
                    with open(tmp_filename, "wb") as fil:
                        pickle.dump(res, fil, pickle.HIGHEST_PROTOCOL)  

                    train_dir.write('{}:{}\n'.format(ind, tmp_filename))
                    tmp_endfile = tmp_filename
                    data_idx += 1

            if comm.rank == 0:
                self.parent.logfile.write(': ~{}\n'.format(tmp_endfile))

        if comm.rank == 0:
            train_dir.close()


    def _parse_strlist(self):
        structures = []
        structure_ind = []
        structure_names = ["None"]
        structure_weights = [1.0]
        name = "None"
        weight = 1.0
        with open(self.structure_list, 'r') as fil:
            for line in fil:
                line = line.strip()
                if len(line) == 0 or line.isspace():
                    name = "None"
                    weight = 1.0
                    continue
                if line[0] == "[" and line[-1] == "]":
                    tmp = line[1:-1]
                    if ':' in tmp:
                        sp = tmp.rsplit(':', 1)
                        try:
                            weight = float(sp[1].strip())
                            name = sp[0].strip()
                        except ValueError:
                            name = tmp.strip()
                            weight = 1.0
                    else:
                        name = tmp.strip()
                        weight = 1.0
                    # If the same structure tags are given multiple times with different weights,
                    # other than first value will be ignored!
                    # Validate structure weight (structure weight is not validated on training run).
                    if name not in structure_names:
                        structure_names.append(name)
                        structure_weights.append(weight)
                        if weight < 0:
                            err = "Structure weight must be greater than or equal to zero."
                            self.parent.logfile.write("Error: {:}\n".format(err))
                            raise ValueError(err)
                        if np.isclose(weight, 0):
                            self.parent.logfile.write("Warning: Structure weight for '{:}' is set to zero.\n".format(name))
                    old_weight = structure_weights[structure_names.index(name)]
                    if not np.isclose(old_weight - weight, 0):
                        self.parent.logfile.write("Warning: Structure weight for '{:}' is set to {:} (previously set to {:}). New value will be ignored\n".format(name, weight, old_weight))
                    continue

                tmp_split = line.split()
                for item in list(braceexpand(tmp_split[0])):
                    structures.append([item, tmp_split[1]])
                    structure_ind.append(structure_names.index(name))

        return structures, structure_ind, structure_names, structure_weights
