from madml import zeros, tensor
from madml.nn import Module, graph_mgr

norm_params = {
    'num_features': -1,
    'eps': 1e-5,
    'momentum': 0.1,
    'axis': -1
}


class normBase(Module):
    def __init__(self, num_features, eps, momentum, axis, affine: bool = True,
                 track_running_stats: bool = True):
        params = norm_params.copy()
        params['num_features'] = num_features
        params['eps'] = eps
        params['momentum'] = momentum
        params['axis'] = axis
        super(normBase, self).__init__(params)

    def forward(self, x: tensor):
        mod = graph_mgr.register('norm', self.name, self.params, {'x': x})
        return self.get_output()

    def get_output(self):
        x = self.get_input('x')
        return self.register_output('y', zeros(x.shape))


class BatchNorm(normBase):
    def __init__(self, num_features=norm_params['num_features'], eps=norm_params['eps'],
                 momentum=norm_params['momentum'], affine: bool = True,
                 track_running_stats: bool = True):
        super(BatchNorm, self).__init__(num_features, eps, momentum, 0, affine, track_running_stats)


class InstanceNorm(normBase):
    def __init__(self, num_features=norm_params['num_features'], eps=norm_params['eps'],
                 momentum=norm_params['momentum'], affine: bool = False,
                 track_running_stats: bool = True):
        super(InstanceNorm, self).__init__(num_features, eps, momentum, 1, affine, track_running_stats)


class LayerNorm(normBase):
    def __init__(self, num_features=norm_params['num_features'], eps=norm_params['eps'],
                 momentum=norm_params['momentum'], affine: bool = True,
                 track_running_stats: bool = True):
        super(LayerNorm, self).__init__(num_features, eps, momentum, -1, affine, track_running_stats)


class GlobalNorm(normBase):
    def __init__(self, num_features=norm_params['num_features'], eps=norm_params['eps'],
                 momentum=norm_params['momentum'], affine: bool = True,
                 track_running_stats: bool = True):
        super(GlobalNorm, self).__init__(num_features, eps, momentum, -2, affine, track_running_stats)
