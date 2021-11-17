from madml import zeros
from madml.nn import Module, graph_mgr

linear_params = {
    'in_feats': 0,
    'out_feats': 0
}


class linear(Module):
    def __init__(self, in_feats=linear_params['in_feats'],
                 out_feats=linear_params['out_feats']):
        params = linear_params.copy()
        params['in_feats'] = in_feats
        params['out_feats'] = out_feats
        super(linear, self).__init__(params)

    def forward(self, x):
        mod = graph_mgr.register('linear', self.name, self.params, {'x': x})
        return self.get_output()

    def get_output(self):
        x = self.get_input('x')
        return self.register_output('y', zeros(x.shape))


graph_mgr.register_module('linear', linear, linear_params)
