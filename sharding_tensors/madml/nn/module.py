from collections import defaultdict

from madml import tensor
from madml.manager import graph_mgr

type_ids = defaultdict(int)


class Parameter(tensor):
    def __init__(self, data, shape):
        super(Parameter, self).__init__(data, shape)


class Module(object):
    type_ids: dict = {}

    def __init__(self, params=None):
        self.params = params
        self.type = str(type(self).__name__)

        if self.type not in Module.type_ids:
            Module.type_ids[self.type] = 0
        Module.type_ids[self.type] += 1
        _id = Module.type_ids[self.type]
        self.name = self.type + '_' + str(_id)

    def forward(self, *args, **kwargs):
        pass

    def backward(self):
        pass

    def get_output(self):
        return graph_mgr.get_output(self, 'y')

    def get_input(self, name):
        return graph_mgr.get_input(self, name)

    def __call__(self, *args, **kwargs):
        y = self.forward(*args, **kwargs)
        return y

    def register_input(self, name, t):
        return graph_mgr.set_input(self, name, t)

    def register_output(self, name, t):
        return graph_mgr.set_output(self, name, t)

    def __str__(self):
        for k, v in graph_mgr.operator_dict.items():
            print(k, '\t', v)
        print('===')
        for k, v in graph_mgr.operations_dict.items():
            print(k, '\t', v)
        print('===')
        return str(graph_mgr.operations_dict)

    def register_dependency(self, x: tensor, y: tensor):
        y.coherency_axis = x.coherency_axis
        return y.coherency_axis
