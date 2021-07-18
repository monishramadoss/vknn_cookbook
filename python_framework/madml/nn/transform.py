from madml import zeros
from madml.nn import Module, graph_mgr


class flatten(Module):
    def __init__(self):
        super(flatten, self).__init__()

    def forward(self, x):
        mod = graph_mgr.register('flatten', self.name, self.params, {'x': x})
        return self.get_output()

    def get_output(self):
        x = self.get_input('x')
        return self.register_output('y', zeros([x.shape[0], x.size(1)]))


graph_mgr.register_module('flatten', flatten, {})
