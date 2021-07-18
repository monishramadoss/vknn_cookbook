from madml import zeros
from madml.nn import Module, graph_mgr

pool_params = {
    'kernel_size': [1],
    'padding': [0],
    'stride': [1],
    'dilation': [1],
}


class maxpoolnd(Module):
    def __init__(self, kernel_size, stride, padding, dilation):
        params = pool_params.copy()
        params['kernel_size'] = kernel_size
        params['padding'] = padding
        params['stride'] = stride
        params['dilation'] = dilation
        super(maxpoolnd, self).__init__(params)

    def forward(self, x):
        mod = graph_mgr.register('maxpool', self.name, self.params, {'x': x})
        return self.get_output()

    def get_output(self):
        x = self.get_input('x')
        return self.register_output('y', zeros(x.shape))


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
