import tensorflow as tf
import numpy as np
import random
import six
from six.moves import cPickle as pickle

"""
Neural network model with symmetry function as a descriptor
"""

def pickle_load(filename):
    with open(filename, 'rb') as fil:
        if six.PY2:
            return pickle.load(fil)
        elif six.PY3:
            return pickle.load(fil, encoding='latin1')

# TODO: complete the code
# TODO: add the part for selecting the memory device(CPU or GPU)
class Neural_network(object):
    def __init__(self):
        self.parent = None
        self.default_inputs = {'neural_network':
                                  {
                                      'method': 'Adam',
                                      'total_epoch': 10000,
                                      'loss_scale': 1.,
                                      'learning_rate': {},
                                      'data_list': ['+./train_dir']
                                  }
                              }

    def _make_filelist(self, filelist):
        # TODO: check tf.data
        #self.filelist = 
        class iterfile(object):
            def __init__(self, filelist):
                self.items = filelist
                self.index = 0

            def __iter__(self):
                return self
            
            def _next(self):
                if self.index == 0:
                    random.shuffle(self.items)
        
                n = self.items[self.index]
                self.index += 1
                self.index = self.index % len(self.items)
                return n

            if six.PY2:
                def next(self):
                    return self._next()
            elif six.PY3:
                def __next__(self):
                    return self._next()

        self.filelist = iterfile(filelist)

    def _get_batch(self, batch_size, initial=False):
        # FIXME: change the batch dict keys to the keys in feed_dict
        self.batch = {'x': dict(),
                      'dx': dict(),
                      'E': list(),
                      'F': list(),
                      'N': dict(),
                      'seg_id': dict()
                      }

        for item in self.parent.inputs['atom_types']:
            self.batch['x'][item] = list()
            self.batch['dx'][item] = list()
            self.batch['N'][item] = list()
        
        for i,item in enumerate(self.filelist):
            loaded_fil = pickle_load(item)

            # TODO: add parameter check part
            self.batch['E'].append(loaded_fil['E'])
            self.batch['F'].append(loaded_fil['F'])
            for jtem in self.parent.inputs['atom_types']:
                self.batch['x'][jtem].append(loaded_fil['x'][jtem])
                self.batch['dx'][jtem].append(loaded_fil['dx'][jtem])
                self.batch['N'][jtem].append(loaded_fil['N'][jtem])

            if i+2 > batch_size:
                break

        self.batch['E'] = np.array(self.batch['E'], dtype=np.float64)
        self.batch['F'] = np.concatenate(self.batch['F']).astype(np.float64)

        atom_num_per_structure = np.sum(list(self.batch['N'].values()), axis=0)
        max_atom_num = np.max(atom_num_per_structure)
        total_atom_num = np.sum(atom_num_per_structure)

        if initial:
            self.inp_size = dict()

        for item in self.parent.inputs['atom_types']:
            self.batch['N'][item] = np.array(self.batch['N'][item], dtype=np.int)
            self.batch['x'][item] = np.array(self.batch['x'][item], dtype=np.float64)

            if initial:
                self.inp_size[item] = self.batch['x'][item].shape[1]
            else:
                if self.inp_size[item] != self.batch['x'][item].shape[1]:
                    raise ValueError

            tmp_dx = np.zeros([total_atom_num, self.inp_size[item],\
                               max_atom_num, 3], dtype=np.float64)

            tmp_idx = 0
            for jtem in self.batch['dx'][item]:
                tmp_dx[tmp_idx:tmp_idx+jtem.shape[0],:,\
                       :jtem.shape[2],:] = jtem
                tmp_idx += jtem.shape[0]
            self.batch['dx'][item] = tmp_idx

            self.batch['seg_id'][item] = \
                np.concatenate([[i]*item for i,item in enumerate(self.batch['N'][item])])
            
            # TODO: scale

        self.batch['dypart'] = \
            np.concatenate([[1]*item + [0]*(total_atom_num - item) for item in atom_num_per_structure])
        

    def _set_scale_parameter(self, scale_file, gdf_file=None):
        self.scale = pickle_load(scale_file)
        # TODO: add the check code for valid scale file
        self.gdf = pickle_load(gdf_file)

    def _make_model(self, inputs, calc_deriv=False):
        self.models = dict()
        self.ys = dict()
        self.dys = dict()
        dtype = self.parent.inputs['dtype']

        for item in self.parent.inputs['atom_types']:
            nodes = self.parent.inputs['nodes'][item]
            nlayers = len(nodes)
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Dense(nodes[0], activation='sigmoid', \
                                            input_dim=self.inp_size[item], dtype=dtype))

            for i in range(1, nlayers):
                model.add(tf.keras.layers.Dense(nodes[i], activation='sigmoid', dtype=dtype))
            model.add(tf.keras.layers.Dense(1, activation='linear', dtype=dtype))

            self.models[item] = model
            self.ys[item] = self.models[item](inputs[item])

            if calc_deriv:
                self.dys[item] = tf.gradients(self.ys[item], inputs[item])[0]
            else:
                self.dys[item] = None


    def _calc_output(self):
        return 0

def _calc_output(atom_types, ys, segsum_ids, \
                 calc_force=False, dys=None, dsyms=None, partition_id=None):
    energy = force = 0
    
    for item in atom_types:
        energy += tf.segment_sum(ys[item], segsum_ids[item])

        if calc_force:
            tmp_force = dsyms[item] * \
                        tf.expand_dims(\
                            tf.expand_dims(dys[item], axis=2), 
                            axis=3)
            tmp_force = tf.reduce_sum(\
                            tf.segment_sum(tmp_force, segsum_ids[item]),
                            axis=1)
            force -= tf.dynamic_partition(tf.reshape(tmp_force, [-1, 3]), \
                                          partition_id, 2)[0]

    return energy, force

def _get_loss(ref_energy, calced_energy, atom_num, \
              calc_force=False, ref_force=None, calced_force=None, force_coeff=0.03, \
              use_gdf=False, gdf_values=None):

    e_loss = tf.reduce_mean(tf.square(calced_energy - ref_energy) / atom_num)

    if calc_force:
        f_loss = tf.square(calced_force - ref_force)
        if use_gdf:
            f_loss *= gdf_values
        f_loss = tf.reduce_mean(f_loss) * force_coeff

    return e_loss, f_loss

def _make_optimizer(loss, method='Adam', lossscale=1., **kwargs):
    if method == 'L-BFGS-B':
        optim = tf.contrib.opt.ScipyOptimizerInterface(lossscale*loss, method=method, options=kwargs)
    elif method == 'Adam':
        learning_rate = tf.train.exponential_decay(**kwargs)
        optim = tf.train.AdamOptimizer(learning_rate=learning_rate, name='Adam').minimize(loss*lossscale)

    return optim

def run(inputs):
    # FIXME: change the code that compatable for other part
    atom_types = inputs['atom_types']

    # read data?
    # preprocessing: scale, GDF...

    # Generate placeholder
    for item in atom_types:
        xs = tf.placeholder(tf.float64, [None, inputs['inp_size'][item]])

    models, ys, dys = _make_model(atom_types, xs, inputs['nodes'], tf.float64, calc_deriv=True)
    energy, force = _calc_output(atom_types, ys, inputs['segid'])
    e_loss, f_loss = _get_loss(inputs['E'], energy, inputs['atnum'])
    optim = _make_optimizer(e_loss + f_loss)

    return 0
    