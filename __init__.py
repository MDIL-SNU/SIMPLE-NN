# import 

class simple_nn(object):
    def __init__(self, inputs, descriptor, model):
        self.inputs = dict()
        self.descriptor = descriptor
        self.model = model

        self.inputs.update(self.descriptor.default_inputs)
        self.inputs.update(self.model.default_inputs)
        self.inputs.update(inputs)

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
        descriptor.inputs = self.inputs
        self._descriptor = descriptor

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        model.parent = self
        model.inputs = self.inputs
        self._model = model


    #def log(self, message):
    #    self._log.write(message)

    def train(self):
        return 0