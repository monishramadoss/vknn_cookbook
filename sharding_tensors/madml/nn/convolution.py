from madml import zeros, tensor
from madml.nn import Module, graph_mgr

conv_params = {
    'filters': -1,
    'kernel_size': [3],
    'padding': [0],
    'stride': [1],
    'dilation': [1],
    'groups': 0
}


def calc_colout_dim(params, x, i):
    col = x.shape[i] + 2 * params['padding'][i] - params['dilation'][i] * (params['kernel_size'][i]-1) - 1
    col = col // params['stride'][-i] + 1
    return col


class convND(Module):
    def __init__(self, filters, kernel_size=conv_params['kernel_size'],
                 stride=conv_params['stride'],
                 padding=conv_params['padding'],
                 dilation=conv_params['dilation'],
                 groups=conv_params['groups']):
        params = conv_params.copy()
        params['filters'] = filters
        params['kernel_size'] = kernel_size
        params['padding'] = padding
        params['stride'] = stride
        params['dilation'] = dilation
        params['groups'] = groups
        super(convND, self).__init__(params)

    def forward(self, x):
        mod = graph_mgr.register('conv', self.name, self.params, {'x': x})
        mod = graph_mgr.register('weight', self.name, self.params, {'w', [self.params['filters']]})
        return self.get_output()

    def get_output(self):
        x = self.get_input('x')
        _col = [x.shape[0], self.params['filters']] +\
               [calc_colout_dim(self.params, x, i) for i in range(x.ndims-1, 2,-1)]
        _vol = x.shape
        col = x.vol2col(_vol, _col, **self.params)
        y = col # @ weight
        return self.register_output('y', col)

    def register_dependency(self, x: tensor, y: tensor):
        y.coherency_axis = x.coherency_axis
        # needs channel to channel mechanism
        # view(0:4 -> 0:19)
        return y.coherency_axis


class conv1d(convND):
    def __init__(self, filters, kernel_size=conv_params['kernel_size'],
                 stride=conv_params['stride'],
                 padding=conv_params['padding'],
                 dilation=conv_params['dilation'],
                 groups=conv_params['groups']):
        params = conv_params.copy()
        params['filters'] = filters
        params['kernel_size'] = kernel_size
        params['padding'] = padding
        params['stride'] = stride
        params['dilation'] = dilation
        params['groups'] = groups
        super(conv1d, self).__init__(**params)


class conv2d(convND):
    def __init__(self, filters, kernel_size=conv_params['kernel_size'],
                 stride=conv_params['stride'],
                 padding=conv_params['padding'],
                 dilation=conv_params['dilation'],
                 groups=conv_params['groups']):
        params = conv_params.copy()
        params['filters'] = filters
        params['kernel_size'] = kernel_size
        params['padding'] = padding
        params['stride'] = stride
        params['dilation'] = dilation
        params['groups'] = groups
        super(conv2d, self).__init__(**params)


class conv3d(convND):
    def __init__(self, filters, kernel_size=conv_params['kernel_size'],
                 stride=conv_params['stride'],
                 padding=conv_params['padding'],
                 dilation=conv_params['dilation'],
                 groups=conv_params['groups']):
        params = conv_params.copy()
        params['filters'] = filters
        params['kernel_size'] = kernel_size
        params['padding'] = padding
        params['stride'] = stride
        params['dilation'] = dilation
        params['groups'] = groups
        super(conv3d, self).__init__(**params)


graph_mgr.register_module('conv', convND, conv_params)


graph.register_moduel('helpConv', convND, )
