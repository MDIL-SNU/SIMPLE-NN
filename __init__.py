import yaml 

# TODO: logging

class simple_nn(object):
    def __init__(self, inputs, descriptor, model):
        # inputs: filename which contains YAML style input parameters
        # descriptor, model
        self.inputs = dict()
        self.descriptor = descriptor
        self.model = model

        self.inputs.update(self.descriptor.default_inputs)
        self.inputs.update(self.model.default_inputs)
        self.inputs.update(yaml.load(open(inputs)))

        if not 'atom_types' in self.inputs:
            raise KeyError
        
        #self.log

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
    def generate_descriptor(self):
        self.descriptor.generate()

    def train(self):
        self.model.train()

    def run(self):
        self.generate_descriptor()
        self.train()