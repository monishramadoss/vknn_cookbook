from madml import zeros
from madml.nn import Module, graph_mgr

conv_params = {
    'filters': 0,
    'kernel_size': [1],
    'padding': [0],
    'stride': [1],
    'dilation': [1],
}


class convnd(Module):
    def __init__(self, filters, kernel_size, stride, padding, dilation):
        params = conv_params.copy()
        params['filters'] = filters
        params['kernel_size'] = kernel_size
        params['padding'] = padding
        params['stride'] = stride
        params['dilation'] = dilation
        super(convnd, self).__init__(params)

    def forward(self, x):
        mod = graph_mgr.register('conv', self.name, self.params, {'x': x})
        return self.get_output()

    def get_output(self):
        x = self.get_input('x')
        return self.register_output('y', zeros(x.shape))


class conv1d(convnd):
    def __init__(self, filters, kernel_size=conv_params['kernel_size'],
                 stride=conv_params['stride'],
                 padding=conv_params['padding'],
                 dilation=conv_params['dilation']):
        params = conv_params.copy()
        params['filters'] = filters
        params['kernel_size'] = kernel_size
        params['padding'] = padding
        params['stride'] = stride
        params['dilation'] = dilation
        super(conv1d, self).__init__(**params)


class conv2d(convnd):
    def __init__(self, filters, kernel_size=conv_params['kernel_size'],
                 stride=conv_params['stride'],
                 padding=conv_params['padding'],
                 dilation=conv_params['dilation']):
        params = conv_params.copy()
        params['filters'] = filters
        params['kernel_size'] = kernel_size
        params['padding'] = padding
        params['stride'] = stride
        params['dilation'] = dilation
        super(conv2d, self).__init__(**params)


class conv3d(convnd):
    def __init__(self, filters, kernel_size=conv_params['kernel_size'],
                 stride=conv_params['stride'],
                 padding=conv_params['padding'],
                 dilation=conv_params['dilation']):
        params = conv_params.copy()
        params['filters'] = filters
        params['kernel_size'] = kernel_size
        params['padding'] = padding
        params['stride'] = stride
        params['dilation'] = dilation
        super(conv3d, self).__init__(**params)


graph_mgr.register_module('conv', convnd, conv_params)
