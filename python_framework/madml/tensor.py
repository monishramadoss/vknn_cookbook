from madml.manager import graph_mgr
from functools import reduce


# todo  introduce context managers

class tensor(object):
    def __init__(self, data, shape=None):
        if shape is None:
            shape = []
        self.data = data
        self.input_layer_name = None
        self.shape = shape

    def __str__(self):
        return self.input_layer_name + ' ' + str(self.shape)

    def __add__(self, other):
        mod = graph_mgr.register('add', self.input_layer_name, inputs={'x': self, 'w': other})
        return mod.get_output()

    def __sub__(self, other):
        mod = graph_mgr.register('sub', self.input_layer_name, inputs={'x': self, 'w': other})
        return mod.get_output()

    def __mul__(self, other):
        mod = graph_mgr.register('mul', self.input_layer_name, inputs={'x': self, 'w': other})
        return mod.get_output()

    def __truediv__(self, other):
        mod = graph_mgr.register('div', self.input_layer_name, inputs={'x': self, 'w': other})
        return mod.get_output()

    def flatten(self):
        mod = graph_mgr.register('flatten', self.input_layer_name, inputs={'x': self})
        return mod.get_output()

    def __matmul__(self, other):
        mod = graph_mgr.register('matmul', self.input_layer_name, inputs={'x': self, 'w': other})
        return mod.get_output()

    def __abs__(self):
        mod = graph_mgr.register('abs', self.input_layer_name, inputs={'x': self})
        return mod.get_output()

    def __mod__(self, other):
        mod = graph_mgr.register('modulo', self.input_layer_name, inputs={'x': self, 'w': other})
        return mod.get_output()

    def __pow__(self, power, modulo=None):
        mod = graph_mgr.register('pow', self.input_layer_name, inputs={'x': self, 'w': power})
        return mod.get_output()

    def __xor__(self, other):
        mod = graph_mgr.register('xor', self.input_layer_name, inputs={'x': self, 'w': other})
        return mod.get_output()

    def __and__(self, other):
        mod = graph_mgr.register('and', self.input_layer_name, inputs={'x': self, 'w': other})
        return mod.get_output()

    def __or__(self, other):
        mod = graph_mgr.register('or', self.input_layer_name, inputs={'x': self, 'w': other})
        return mod.get_output()

    def __neg__(self):
        mod = graph_mgr.register('neg', self.input_layer_name, inputs={'x': self})
        return mod.get_output()

    def __eq__(self, other):
        mod = graph_mgr.register('eq', self.input_layer_name, inputs={'x': self, 'w': other})
        return mod.get_output()

    def __ne__(self, other):
        mod = graph_mgr.register('ne', self.input_layer_name, inputs={'x': self, 'w': other})
        return mod.get_output()

    def __ge__(self, other):
        mod = graph_mgr.register('ge', self.input_layer_name, inputs={'x': self, 'w': other})
        return mod.get_output()

    def __le__(self, other):
        mod = graph_mgr.register('le', self.input_layer_name, inputs={'x': self, 'w': other})
        return mod.get_output()

    def __gt__(self, other):
        mod = graph_mgr.register('gt', self.input_layer_name, inputs={'x': self, 'w': other})
        return mod.get_output()

    def __lt__(self, other):
        mod = graph_mgr.register('lt', self.input_layer_name, inputs={'x': self, 'w': other})
        return mod.get_output()

    def backward(self):
        graph_mgr.backward()

    def size(self, idx=0):
        return reduce(lambda x, y: x * y, self.shape[idx:])
