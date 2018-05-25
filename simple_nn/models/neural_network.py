import tensorflow as tf
import numpy as np
import random
import six
from six.moves import cPickle as pickle
import collections
from ..utils import _make_data_list, pickle_load, preprocessing, _generate_gdf_file

"""
Neural network model with symmetry function as a descriptor
"""

# TODO: add the part for selecting the memory device(CPU or GPU)
# TODO: BFGS support
# TODO: add regularization
class Neural_network(object):
    def __init__(self):
        self.parent = None
        self.key = 'neural_network'
        self.default_inputs = {'neural_network':
                                  {
                                      'method': 'Adam',
                                      'continue': False,
                                      'use_force': False,
                                      'force_coeff': 0.3,
                                      'total_epoch': 10000,
                                      'save_interval': 1000,
                                      'show_interval': 100,
                                      'max_iteration': 1000,
                                      'batch_size': 64,
                                      'loss_scale': 1.,
                                      'double_precision': True,
                                      'valid_rate': 0.1,
                                      'generate_valid': True,
                                      'scale': True,
                                      'learning_rate': 0.01,
                                      'optimizer': dict(),
                                      'nodes': '30-30',
                                      'test': False,
                                      'train': True,
                                      'atomic_weights': {
                                          'type': None,
                                          'params': dict(),
                                      },
                                      'regularization': {
                                          'type': None,
                                          'params': dict(),
                                      },
                                  }
                              }
        self.inputs = dict()
        self.global_step = tf.Variable(0, trainable=False)
        self.train_data_list = './train_list'
        self.valid_data_list = './valid_list'
        self.test_data_list = './test_list'

    def _make_fileiter(self):
        # TODO: check tf.data
        if self.inputs['generate_valid']:
            with open(self.train_data_list, 'r') as fil:
                full_data = fil.readlines()

            random.shuffle(full_data)
            valid_idx = int(len(full_data) * self.inputs['valid_rate'])

            with open(self.valid_data_list, 'w') as fil:
                for item in full_data[:valid_idx]:
                    fil.write(item)

            with open(self.train_data_list, 'w') as fil:
                for item in full_data[valid_idx:]:
                    fil.write(item)

            self.inputs['generate_valid'] = False
            self.parent.write_inputs()

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

        if self.inputs['train']:
            train_data = _make_data_list(self.train_data_list)
            train_data = list(zip(train_data, list(range(len(train_data)))))
            train_data = iterfile(train_data)
            valid_data = _make_data_list(self.valid_data_list)
            valid_data = list(zip(valid_data, list(range(len(valid_data)))))
        else:
            train_data = valid_data = None

        if self.inputs['test']:
            test_data = _make_data_list(self.test_data_list)
            test_data = list(zip(test_data, list(range(len(test_data)))))
        else:
            test_data = None
 
        return train_data, valid_data, test_data

    def _get_batch(self, fileiter, batch_size, initial=False, valid=False):
        batch = {
            'x': dict(),
            'dx': dict(),
            '_E': list(),
            '_F': list(),
            'N': dict(),
            'seg_id': dict()
        }
        
        tag_atomic_weights = self.inputs['atomic_weights']['type']
        if tag_atomic_weights != None:
            batch['atomic_weights'] = list()

        for item in self.parent.inputs['atom_types']:
            batch['x'][item] = list()
            batch['dx'][item] = list()
            batch['N'][item] = list()
        
        for i,item in enumerate(fileiter):
            loaded_fil = pickle_load(item[0])

            # TODO: add parameter check part
            batch['_E'].append(loaded_fil['E'])
            batch['_F'].append(loaded_fil['F'])
            for jtem in self.parent.inputs['atom_types']:
                batch['x'][jtem].append(loaded_fil['x'][jtem])
                batch['dx'][jtem].append(loaded_fil['dx'][jtem])
                batch['N'][jtem].append(loaded_fil['N'][jtem])
                if tag_atomic_weights != None:
                    if valid:
                        batch['atomic_weights'].append([1.]*loaded_fil['N'][jtem])
                    else:
                        batch['atomic_weights'].\
                            append(self.atomic_weights_full[jtem][self.atomic_weights_full[jtem][:,1] == item[1],1])

            if initial:
                self.params = loaded_fil['params']

            if (not valid) and (i+2 > batch_size):
                break

        batch['_E'] = np.array(batch['_E'], dtype=np.float64).reshape([-1,1])
        batch['_F'] = np.concatenate(batch['_F']).astype(np.float64)
        if tag_atomic_weights != None:
            batch['atomic_weights'] = np.concatenate(batch['atomic_weights']).astype(np.float64).reshape([-1,1])

        batch['tot_num'] = np.sum(list(batch['N'].values()), axis=0)
        max_atom_num = np.max(batch['tot_num'])

        if initial:
            self.inp_size = dict()

        for item in self.parent.inputs['atom_types']:
            batch['N'][item] = np.array(batch['N'][item], dtype=np.int)
            batch['x'][item] = np.concatenate(batch['x'][item], axis=0).astype(np.float64)
            batch['x'][item] -= self.scale[item][0:1,:]
            batch['x'][item] /= self.scale[item][1:2,:]

            if initial:
                self.inp_size[item] = batch['x'][item].shape[1]
            else:
                if self.inp_size[item] != batch['x'][item].shape[1]:
                    raise ValueError

            tmp_dx = np.zeros([np.sum(batch['N'][item]), self.inp_size[item],\
                               max_atom_num, 3], dtype=np.float64)

            tmp_idx = 0
            for jtem in batch['dx'][item]:
                tmp_dx[tmp_idx:tmp_idx+jtem.shape[0],:,:jtem.shape[2],:] = jtem
                tmp_idx += jtem.shape[0]
            batch['dx'][item] = tmp_dx / self.scale[item][1:2,:].reshape([1,self.inp_size[item],1,1])

            batch['seg_id'][item] = \
                np.concatenate([[j]*jtem for j,jtem in enumerate(batch['N'][item])])

        batch['partition'] = \
            np.concatenate([[0]*item + [1]*(max_atom_num - item) for item in batch['tot_num']])
        
        return batch

    def _set_scale_parameter(self, scale_file, gdf_file=None):
        self.scale = pickle_load(scale_file)
        # TODO: add the check code for valid scale file
        self.gdf = pickle_load(gdf_file)

    def _make_model(self):
        self.models = dict()
        self.ys = dict()
        self.dys = dict()

        if self.inputs['double_precision']:
            dtype = tf.float64
        else:
            dtype = tf.float32

        dense_basic_setting = {
            'dtype': dtype,
            'kernel_initializer': tf.initializers.truncated_normal(stddev=0.3, dtype=dtype),
            'bias_initializer': tf.initializers.truncated_normal(stddev=0.3, dtype=dtype)
        }

        self.nodes = dict()
        for item in self.parent.inputs['atom_types']:
            if isinstance(self.inputs['nodes'], collections.Mapping):
                nodes = list(map(int, self.inputs['nodes'][item].split('-')))
            else:
                nodes = list(map(int, self.inputs['nodes'].split('-')))
            nlayers = len(nodes)
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Dense(nodes[0], activation='sigmoid', \
                                            input_dim=self.inp_size[item],
                                            **dense_basic_setting))

            for i in range(1, nlayers):
                model.add(tf.keras.layers.Dense(nodes[i], activation='sigmoid', **dense_basic_setting))
            model.add(tf.keras.layers.Dense(1, activation='linear', **dense_basic_setting))

            nodes.append(1)
            self.nodes[item] = nodes

            self.models[item] = model
            self.ys[item] = self.models[item](self.x[item])

            if self.inputs['use_force']:
                self.dys[item] = tf.gradients(self.ys[item], self.x[item])[0]
            else:
                self.dys[item] = None


    def _calc_output(self):
        self.E = self.F = 0

        for item in self.parent.inputs['atom_types']:
            self.E += tf.segment_sum(self.ys[item], self.seg_id[item])

            if self.inputs['use_force']:
                tmp_force = self.dx[item] * \
                            tf.expand_dims(\
                                tf.expand_dims(self.dys[item], axis=2),
                                axis=3)
                tmp_force = tf.reduce_sum(\
                                tf.segment_sum(tmp_force, self.seg_id[item]),
                                axis=1)
                self.F -= tf.dynamic_partition(tf.reshape(tmp_force, [-1,3]),
                                                   self.partition, 2)[0]

    def _get_loss(self, use_gdf=False, atomic_weights=None):
        self.e_loss = tf.reduce_mean(tf.square((self._E - self.E) / self.tot_num))
        self.total_loss = self.e_loss

        if self.inputs['use_force']:
            self.f_loss = tf.square(self._F - self.F)
            if self.inputs['atomic_weights']['type'] is not None:
                self.f_loss *= self.atomic_weights
            self.f_loss = tf.reduce_mean(self.f_loss) * self.inputs['force_coeff']
        
            self.total_loss += self.f_loss

        if self.inputs['regularization']['type'] is not None:
            if self.inputs['regularization']['type'] == 'l2':
                coeff = self.inputs['regularization']['params'].get('coeff', 1e-6)
                self.reg_loss = 0
                for item in self.parent.inputs['atom_types']:
                    for weight in self.models[item].weights:
                        self.reg_loss += tf.nn.l2_loss(weight)
                self.total_loss += self.reg_loss * coeff
            else:
                raise NotImplementedError

    def _make_optimizer(self, user_optimizer=None):
        final_loss = self.inputs['loss_scale']*self.total_loss
        if self.inputs['method'] == 'L-BFGS-B':
            self.optim = tf.contrib.opt.ScipyOptimizerInterface(final_loss, 
                                                                method=self.inputs['method'], 
                                                                options=self.inputs['optimizer'])
        elif self.inputs['method'] == 'Adam':
            if isinstance(self.inputs['learning_rate'], collections.Mapping):
                self.learning_rate = tf.train.exponential_decay(global_step=self.global_step, **self.inputs['learning_rate'])
            else:
                self.learning_rate = tf.constant(self.inputs['learning_rate'])

            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate, 
                                                name='Adam', **self.inputs['optimizer'])
            self.optim = self.optim.minimize(final_loss, global_step=self.global_step)
        else:
            if user_optimizer != None:
                self.optim = user_optimizer.minimize(final_loss, global_step=self.global_step)
            else:
                raise ValueError

    def _make_feed_dict(self, batch):
        fdict = {
            self._E: batch['_E'],
            self._F: batch['_F'],
            self.tot_num: batch['tot_num'],
            self.partition: batch['partition']
        }

        if self.inputs['atomic_weights']['type'] != None:
            fdict[self.atomic_weights] = batch['atomic_weights']

        for item in self.parent.inputs['atom_types']:
            fdict[self.x[item]] = batch['x'][item]
            fdict[self.dx[item]] = batch['dx'][item]
            fdict[self.seg_id[item]] = batch['seg_id'][item]

        return fdict 

    def _generate_lammps_potential(self, sess):
        # TODO: get the parameter info from initial batch generting processs
        atom_type_str = ' '.join(self.parent.inputs['atom_types'])

        filename = './potential_saved'
        FIL = open(filename, 'w')
        FIL.write('ELEM_LIST ' + atom_type_str + '\n\n')

        for item in self.parent.inputs['atom_types']:
            FIL.write('POT {} {}\n'.format(item, np.max(self.params[item][:,3])))
            FIL.write('SYM {}\n'.format(len(self.params[item])))

            for ctem in self.params[item]:
                tmp_types = self.parent.inputs['atom_types'][int(ctem[1])-1]
                if int(ctem[0]) > 3:
                    tmp_types += ' {}'.format(self.parent.inputs['atom_types'][int(ctem[2])-1])

                FIL.write('{} {} {} {} {} {}\n'.\
                    format(int(ctem[0]), ctem[3], ctem[4], ctem[5], ctem[6], tmp_types))

            FIL.write('scale1 {}\n'.format(' '.join(self.scale[item][0,:].astype(np.str))))
            FIL.write('scale2 {}\n'.format(' '.join(self.scale[item][1,:].astype(np.str))))
            
            weights = sess.run(self.models[item].weights)
            nlayers = len(self.nodes[item])
            FIL.write('NET {} {}\n'.format(nlayers-1, ' '.join(map(str, self.nodes[item]))))

            for j in range(nlayers):
                # FIXME: add activation function type if new activation is added
                if j == nlayers-1:
                    acti = 'linear'
                else:
                    acti = 'sigmoid'

                FIL.write('LAYER {} {}\n'.format(j, acti))

                for k in range(self.nodes[item][j]):
                    FIL.write('w{} {}\n'.format(k, ' '.join(weights[j*2][:,k].astype(np.str))))
                    FIL.write('b{} {}\n'.format(k, weights[j*2 + 1][k]))
            
            FIL.write('\n')

        FIL.close()

    def _save(self, sess, saver):
        if not self.inputs['continue']:
            self.inputs['continue'] = True
            self.parent.write_inputs()

        self.parent.logfile.write("Save the weights and write the LAMMPS potential..\n")              
        saver.save(sess, './SAVER')
        self._generate_lammps_potential(sess)

    def train(self, user_optimizer=None, user_atomic_weights_function=None):
        self.inputs = self.parent.inputs['neural_network']
        # read data?
        
        train_fileiter, valid_fileiter, test_fileiter = self._make_fileiter()

        # preprocessing: scale, GDF...
        if self.inputs['atomic_weights']['type'] == 'gdf':
            get_atomic_weights = _generate_gdf_file
        elif self.inputs['atomic_weights']['type'] == 'user':
            get_atomic_weights = user_atomic_weights_function
        elif self.inputs['atomic_weights']['type'] == 'file':
            get_atomic_weights = './atomic_weights'
        else:
            get_atomic_weights = None

        # FIXME: Error occur: only test
        if self.inputs['train']:
            self.scale, self.atomic_weights_full = \
                preprocessing(self.train_data_list, self.parent.inputs['atom_types'], 'x', \
                            calc_scale=self.inputs['scale'], \
                            get_atomic_weights=get_atomic_weights, \
                            **self.inputs['atomic_weights']['params'])

            self._get_batch(train_fileiter, 1, initial=True)
        else:
            self.scale, self.atomic_weights_full = \
                preprocessing(self.test_data_list, self.parent.inputs['atom_types'], 'x', \
                            calc_scale=False, get_atomic_weights=None)
            self._get_batch(test_fileiter, 1, initial=True, valid=True)

        # Generate placeholder
        self._E = tf.placeholder(tf.float64, [None, 1])
        self._F = tf.placeholder(tf.float64, [None, 3])
        self.tot_num = tf.placeholder(tf.float64, [None])
        self.partition = tf.placeholder(tf.int32, [None])
        self.seg_id = dict()
        self.x = dict()
        self.dx = dict()
        self.atomic_weights = tf.placeholder(tf.float64, [None, 1]) \
                                if self.inputs['atomic_weights']['type'] != None else None
        for item in self.parent.inputs['atom_types']:
            self.x[item] = tf.placeholder(tf.float64, [None, self.inp_size[item]])
            self.dx[item] = tf.placeholder(tf.float64, [None, self.inp_size[item], None, 3])
            self.seg_id[item] = tf.placeholder(tf.int32, [None])

        self._make_model()
        self._calc_output()
        self._get_loss()
        self._make_optimizer(user_optimizer=user_optimizer)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = 0.45
        with tf.Session(config=config) as sess:
            # Load or initialize the variables
            saver = tf.train.Saver()
            if self.inputs['continue']:
                saver.restore(sess, './SAVER')
            else:
                sess.run(tf.global_variables_initializer())

            if self.inputs['train']:
                if self.inputs['method'] == 'L-BFGS-B':
                    # TODO: complete this part
                    raise ValueError
                elif self.inputs['method'] == 'Adam':
                    valid_set = self._get_batch(valid_fileiter, 1, valid=True)
                    valid_fdict = self._make_feed_dict(valid_set)

                    for epoch in range(self.inputs['total_epoch']):
                        train_batch = self._get_batch(train_fileiter, self.inputs['batch_size'])
                        train_fdict = self._make_feed_dict(train_batch)
                        self.optim.run(feed_dict=train_fdict)

                        # Logging
                        if (epoch+1) % self.inputs['show_interval'] == 0:
                            result = "epoch {:7d}: ".format(sess.run(self.global_step)+1)

                            eloss = sess.run(self.e_loss, feed_dict=valid_fdict)
                            eloss = np.sqrt(eloss)
                            result += 'E loss = {:6.4e}'.format(eloss)

                            if self.inputs['use_force']:
                                floss = sess.run(self.f_loss, feed_dict=valid_fdict)
                                floss = np.sqrt(floss*3/self.inputs['force_coeff'])
                                result += ', F loss = {:6.4e}'.format(floss)

                            lr = sess.run(self.learning_rate)
                            result += ', learning_rate: {:6.4e}\n'.format(lr)
                            self.parent.logfile.write(result)

                        # Temp saving
                        if (epoch+1) % self.inputs['save_interval'] == 0:
                            self._save(sess, saver)

                self._save(sess, saver)

            if self.inputs['test']:
                test_set = self._get_batch(test_fileiter, 1, valid=True)
                test_fdict = self._make_feed_dict(test_set)

                test_save = dict()
                test_save['DFT_E'] = test_set['_E']
                test_save['NN_E'] = sess.run(self.E, feed_dict=test_fdict)

                eloss = sess.run(self.e_loss, feed_dict=test_fdict)
                eloss = np.sqrt(eloss)

                if self.inputs['use_force']:
                    test_save['DFT_F'] = test_set['_F']
                    test_save['NN_F'] = sess.run(self.F, feed_dict=test_fdict)

                    floss = sess.run(self.f_loss, feed_dict=test_fdict)
                    floss = np.sqrt(floss*3/self.inputs['force_coeff'])

                with open('./test_result', 'wb') as fil:
                    pickle.dump(test_save, fil, pickle.HIGHEST_PROTOCOL)

                self.parent.logfile.write('Test result saved..\n')
                self.parent.logfile.write(' Test E RMSE: {}, F RMSE: {}\n'.format(eloss, floss))

                
    
