from madml import zeros
from . import Module, graph_mgr


class relu(Module):
    def __init__(self):
        super(relu, self).__init__()

    def forward(self, x):
        graph_mgr.register('relu', self.name, {}, {'x': x})
        return self.get_output()

    def get_output(self):
        x = self.get_input('x')
        return self.register_output('y', zeros(x.shape))


def ReLU(x):
    mod = graph_mgr.register('relu', x.input_layer_name, inputs={'x': x})
    return mod.get_output()


graph_mgr.register_module('relu', relu, {})
