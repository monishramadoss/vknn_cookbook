from madml import zeros, tensor
from madml.nn import Module, graph_mgr

pool_params = {
    'kernel_size': [1],
    'padding': [0],
    'stride': [1],
    'dilation': [1],
}


def calc_colout_dim(params, x, i):
    col = x.shape[i] + 2 * params['padding'][i] - params['dilation'][i] * (params['kernel_size'][i] - 1) - 1
    col = col // params['stride'][-i] + 1
    return col


class maxpoolnd(Module):
    def __init__(self, kernel_size, stride, padding, dilation):
        params = pool_params.copy()
        params['kernel_size'] = kernel_size
        params['padding'] = padding
        params['stride'] = stride
        params['dilation'] = dilation
        super(maxpoolnd, self).__init__(params)

    def forward(self, x: tensor):
        mod = graph_mgr.register('maxpool', self.name, self.params, {'x': x})
        return self.get_output()

    def get_output(self):
        x = self.get_input('x')
        _col = [x.shape[0], x.shape[1]] + \
               [calc_colout_dim(self.params, x, i) for i in range(x.ndims - 1, 2, -1)]
        return self.register_output('y', zeros(_col))

    def register_dependency(self, x: tensor, y: tensor):
        # TODO: link dependency on axes

        return None

class maxpool1d(maxpoolnd):
    def __init__(self, kernel_size=pool_params['kernel_size'],
                 stride=pool_params['stride'],
                 padding=pool_params['padding'],
                 dilation=pool_params['dilation']):
        params = pool_params.copy()
        params['kernel_size'] = kernel_size
        params['padding'] = padding
        params['stride'] = stride
        params['dilation'] = dilation
        super(maxpool1d, self).__init__(**params)


class maxpool2d(maxpoolnd):
    def __init__(self, kernel_size=pool_params['kernel_size'],
                 stride=pool_params['stride'],
                 padding=pool_params['padding'],
                 dilation=pool_params['dilation']):
        params = pool_params.copy()
        params['kernel_size'] = kernel_size
        params['padding'] = padding
        params['stride'] = stride
        params['dilation'] = dilation
        super(maxpool2d, self).__init__(**params)


class maxpool3d(maxpoolnd):
    def __init__(self, kernel_size=pool_params['kernel_size'],
                 stride=pool_params['stride'],
                 padding=pool_params['padding'],
                 dilation=pool_params['dilation']):
        params = pool_params.copy()
        params['kernel_size'] = kernel_size
        params['padding'] = padding
        params['stride'] = stride
        params['dilation'] = dilation
        super(maxpool3d, self).__init__(**params)


graph_mgr.register_module('maxpool', maxpoolnd, pool_params)
