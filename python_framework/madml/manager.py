import os
from collections import OrderedDict


class Manager(object):
    def __init__(self, num_threads=os.cpu_count() // 2):
        self.num_threads = num_threads
        self.operations_dict = {}
        self.operator_dict = {}

    def register_module(self, name, cls, param):
        self.operator_dict[name] = {'obj': cls, 'param': param}

    def register(self, tp, name=None, params={}, inputs={}):
        if name not in self.operations_dict:
            mod_cls = self.operator_dict[tp]['obj']
            mod = mod_cls(**params)
            if name is None:
                name = mod.name
            else:
                mod.name = name
            self.operations_dict[mod.name] = {'module': mod, 'inputs': {}, 'outputs': {}}

        mod = self.operations_dict[name]['module']
        for k in inputs:
            self.set_input(mod, k, inputs[k])
        return mod

    def set_input(self, module, name, tensor):
        self.operations_dict[module.name]['inputs'][name] = tensor
        if tensor.input_layer_name is None:
            tensor.input_layer_name = name
        return tensor

    def set_output(self, module, name, tensor):
        if name not in self.operations_dict[module.name]['outputs']:
            self.operations_dict[module.name]['outputs'][name] = tensor
        return tensor

    def get_input(self, module, name):
        if name not in self.operations_dict[module.name]['inputs']:
            return None
        return self.operations_dict[module.name]['inputs'][name]

    def get_output(self, module, name):
        if name not in self.operations_dict[module.name]['outputs']:
            return None
        return self.operations_dict[module.name]['outputs'][name]

    def get_mod(self, name):
        return self.operations_dict[name]['module']

    def backward(self):
        pass

graph_mgr = Manager()
