import sys
import os
import yaml
import collections
import functools
import atexit
from .utils import modified_sigmoid, _generate_gdf_file
from ._version import __version__, __git_sha__
from .utils.mpiclass import DummyMPI, MPI4PY
import tensorflow as tf
import numpy as np


# TODO: logging

def deep_update(source, overrides, warn_new_key=False, logfile=None, comm=None, depth=0, parent="top"):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.

    :param dict source: base dictionary to be updated
    :param dict overrides: new dictionary
    :param bool warn_new_key: if true, warn about new keys in overrides
    :param str logfile: filename to which warnings are written (if not given warnings are written to stdout)
    :returns: updated dictionary source
    """
    if comm is None:
        comm = DummyMPI()

    if logfile is None:
        logfile = sys.stdout

    for key in overrides.keys():
        if isinstance(source, collections.Mapping):
            if warn_new_key and depth < 2 and key not in source and comm.rank == 0:
                logfile.write("Warning: Unidentified option in {:}: {:}\n".format(parent, key))
            if isinstance(overrides[key], collections.Mapping) and overrides[key]:
                returned = deep_update(source.get(key, {}), overrides[key],
                                       warn_new_key=warn_new_key, logfile=logfile, comm=comm,
                                       depth=depth+1, parent=key)
                source[key] = returned
            # Need list append?
            else:
                source[key] = overrides[key]
        else:
            source = {key: overrides[key]}
    return source

class Simple_nn(object):
    """
    Base class for running simple-nn

    :param str inputs: filename which contains YAML style input parameters
    :param class descriptor: subclass for generating feature vectors
    :param class model: subclass for training machine learning model
    """
    def __init__(self, inputs, descriptor=None, model=None):
        """
        
        """
        self.logfile = sys.stdout
        try:
            import mpi4py
        except ImportError:
            self.comm = DummyMPI()
        else:
            self.comm = MPI4PY()
        if self.comm.rank == 0:
            # Only proc 0 opens and writes to LOG file.
            self.logfile = open('LOG', 'w', 10)
            atexit.register(self._close_log)
            self._log_header()

        self.default_inputs = {
            'generate_features': True,
            'preprocess': False,
            'train_model': True,
            'atom_types': [],
            'random_seed': None,
            }

        self.inputs = self.default_inputs

        if descriptor != None:
            self.descriptor = descriptor
            self.inputs = deep_update(self.inputs, self.descriptor.default_inputs)

        if model != None:
            self.model = model
            self.inputs = deep_update(self.inputs, self.model.default_inputs)

        with open(inputs) as input_file:
            self.inputs = deep_update(self.inputs, yaml.safe_load(input_file), warn_new_key=True,
                                      logfile=self.logfile, comm=self.comm)

        if len(self.inputs['atom_types']) == 0:
            raise KeyError

        if not self.inputs['neural_network']['use_force'] and \
                self.inputs['symmetry_function']['atomic_weights']['type'] is not None:
            self.logfile.write("Warning: Force training is off but atomic weights are given. Atomic weights will be ignored.\n")

        if self.inputs['neural_network']['method'] == 'L-BFGS' and \
                not self.inputs['neural_network']['full_batch']:
            self.logfile.write("Warning: Optimization method is L-BFGS but full batch mode is off. This might results bad convergence or divergence.\n")

        if self.inputs['random_seed'] is not None:
            seed = self.inputs['random_seed']
            tf.set_random_seed(seed)
            np.random.seed(seed)
            self.logfile.write("*** Random seed: {0:} ***\n".format(seed))

    def _close_log(self):
        self.logfile.flush()
        os.fsync(self.logfile.fileno())
        self.logfile.close()

    def _log_header(self):
        # TODO: make the log header (low priority)
        self.logfile.write("SIMPLE_NN v{0:} ({1:})\n".format(__version__, __git_sha__))

    def write_inputs(self):
        """
        Write current input parameters to the 'input_cont.yaml' file
        """
        with open('input_cont.yaml', 'w') as fil:
            yaml.dump(self.inputs, fil, default_flow_style=False)

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    def inputs(self, inputs):
        self._inputs = inputs

    @property
    def descriptor(self):
        return self._descriptor

    @descriptor.setter
    def descriptor(self, descriptor):
        descriptor.parent = self
        self._descriptor = descriptor

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        model.parent = self
        self._model = model


    #def log(self, message):
    #    self._log.write(message)

    def run(self, user_atomic_weights_function=None, user_optimizer=None):
        """
        Function for running simple-nn.

        :param user_optimizer: tensorflow optimizer other than AdamOptimizer. 
        """

        self.descriptor.set_inputs()
        self.model.set_inputs()
        modifier = None
        if self.descriptor.inputs['weight_modifier']['type'] == 'modified sigmoid':
            modifier = dict()
            #modifier = functools.partial(modified_sigmoid, **self.descriptor.inputs['weight_modifier']['params'])
            for item in self.inputs['atom_types']:
                modifier[item] = functools.partial(modified_sigmoid, **self.descriptor.inputs['weight_modifier']['params'][item])
        if self.descriptor.inputs['atomic_weights']['type'] == 'gdf':
            #get_atomic_weights = functools.partial(_generate_gdf_file)#, modifier=modifier)
            get_atomic_weights = _generate_gdf_file
        elif self.descriptor.inputs['atomic_weights']['type'] == 'user':
            get_atomic_weights = user_atomic_weights_function
        elif self.descriptor.inputs['atomic_weights']['type'] == 'file':
            get_atomic_weights = './atomic_weights'
        else:
            get_atomic_weights = None

        if self.inputs['generate_features']:
            self.descriptor.generate()
            self.descriptor.preprocess(use_force=self.inputs['neural_network']['use_force'],
                                       use_stress=self.inputs['neural_network']['use_stress'],
                                       get_atomic_weights=get_atomic_weights,
                                       **self.descriptor.inputs['atomic_weights']['params'])
        elif self.inputs['preprocess']:
            self.descriptor.preprocess(use_force=self.inputs['neural_network']['use_force'], 
                                       use_stress=self.inputs['neural_network']['use_stress'],
                                       get_atomic_weights=get_atomic_weights, 
                                       **self.descriptor.inputs['atomic_weights']['params'])
        
        if self.inputs['train_model']:
            if self.comm.size > 1:
                if self.comm.rank == 0:
                    self.logfile.write("Error: Training model with MPI is not supported! Please restart training without MPI (set generate_features: false, preprocess: false, and train_model: true to run only training).\n")
                sys.exit(0)
            self.model.train(user_optimizer=user_optimizer, aw_modifier=modifier)
