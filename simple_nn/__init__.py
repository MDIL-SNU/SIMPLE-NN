import yaml 
import collections
import functools
from .utils import modified_sigmoid, _generate_gdf_file

# TODO: logging

def deep_update(source, overrides):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.

    :param dict source: base dictionary to be updated
    :param dict overrides: new dictionary
    :returns: updated dictionary source
    """
    for key in overrides.keys():
        if isinstance(source, collections.Mapping):
            if isinstance(overrides[key], collections.Mapping) and overrides[key]:
                returned = deep_update(source.get(key, {}), overrides[key])
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
        self.default_inputs = {
            'generate_features': True,
            'preprocess': False,
            'train_model': True
            }

        self.inputs = self.default_inputs

        if descriptor != None:
            self.descriptor = descriptor
            self.inputs = deep_update(self.inputs, self.descriptor.default_inputs)

        if model != None:
            self.model = model
            self.inputs = deep_update(self.inputs, self.model.default_inputs)

        self.inputs = deep_update(self.inputs, yaml.load(open(inputs)))

        if not 'atom_types' in self.inputs:
            raise KeyError
        
        self.logfile = open('LOG', 'w', 10)

    def _log_header(self):
        # TODO: make the log header (low priority)
        self.logfile.write("SIMPLE_NN\n")

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
        modifier = None
        if self.descriptor.inputs['weight_modifier']['type'] == 'modified sigmoid':
            modifier = functools.partial(modified_sigmoid, **self.inputs['weight_modifier']['params'])
        if self.descriptor.inputs['atomic_weights']['type'] == 'gdf':
            get_atomic_weights = functools.partial(_generate_gdf_file, modifier=modifier)
        elif self.descriptor.inputs['atomic_weights']['type'] == 'user':
            get_atomic_weights = user_atomic_weights_function
        elif self.descriptor.inputs['atomic_weights']['type'] == 'file':
            get_atomic_weights = './atomic_weights'
        else:
            get_atomic_weights = None

        if self.inputs['generate_features']:
            self.descriptor.generate()
            self.descriptor.preprocess(get_atomic_weights=get_atomic_weights,
                                       **self.descriptor.inputs['atomic_weights']['params'])
        elif self.inputs['preprocess']:
            self.descriptor.preprocess(get_atomic_weights=get_atomic_weights, 
                                       **self.descriptor.inputs['atomic_weights']['params'])
        
        if self.inputs['train_model']:
            self.model.train(user_optimizer=user_optimizer)
