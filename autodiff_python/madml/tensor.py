from madml.manager import graph_mgr
from functools import reduce
from collections import defaultdict


# todo  introduce context managers

class tensor(object):
    def __init__(self, data, shape=None, dtype='float32', local_grads=[]):
        if shape is None:
            shape = []
        self.data = data
        self.input_layer_name = ''
        self.shape = shape
        self.layer = ''
        self.ndim = len(shape)
        self.dtype = dtype
        self.local_grads = local_grads

    def __len__(self):
        return sum(self.shape)

    def __str__(self):
        return self.input_layer_name + ' ' + str(self.shape) + ' ' + 'layer_name: ' + self.layer

    def backward(self):
        return get_gradient(self, self.local_grads[0])

    def size(self, idx=0):
        return reduce(lambda x, y: x * y, self.shape[idx:])


def add(a, b):
    local_grads = [
        (a, lambda path_value: path_value),
        (b, lambda path_value: path_value)
    ]
    return tensor([0 for _ in range(len(a))], shape=a.shape, dtype=a.dtype, local_grads=local_grads)


def sub(a, b):
    local_grads = [
        (a, lambda path_value: path_value),
        (b, lambda path_value: -path_value)
    ]
    return tensor([0 for _ in range(len(a))], shape=a.shape, dtype=a.dtype, local_grads=local_grads)


def mul(a, b):
    local_grads = [
        (a, lambda path_value: path_value * b),
        (b, lambda path_value: path_value * a)
    ]
    return tensor([0 for _ in range(len(a))], shape=a.shape, dtype=a.dtype, local_grads=local_grads)


def div(a, b):
    local_grads = [
        (a, lambda path_value: path_value / b),
        (b, lambda path_value: -path_value * a / (b * b))
    ]
    return tensor([0 for _ in range(len(a))], shape=a.shape, dtype=a.dtype, local_grads=local_grads)


def neg(a):
    local_grads = [
        (a, lambda path_value: -path_value)
    ]
    return tensor([0 for _ in range(len(a))], shape=a.shape, dtype=a.dtype, local_grads=local_grads)


def inv(a):
    local_grads = [
        (a, lambda path_value: -1 / (path_value * path_value))
    ]
    return tensor([0 for _ in range(len(a))], shape=a.shape, dtype=a.dtype, local_grads=local_grads)


def exp(a):
    local_grads = [
        (a, lambda path_value: path_value * exp(a))
    ]
    return tensor([0 for _ in range(len(a))], shape=a.shape, dtype=a.dtype, local_grads=local_grads)


def log(a):
    local_grads = [
        (a, lambda path_value: path_value / a)
    ]
    return tensor([0 for _ in range(len(a))], shape=a.shape, dtype=a.dtype, local_grads=local_grads)


def reshape(a, shape):
    local_grads = [
        (a, lambda path_value: path_value.reshape(a.shap))
    ]
    return tensor([0 for _ in range(len(a))], shape=shape, dtype=a.dtype, local_grads=local_grads)


def get_gradient(a: tensor, loss: tensor):
    grad = defaultdict(lambda: tensor(0, [1]))

    def compute_gradients(var: tensor, path_value: tensor):
        for child, loc_grad in var.local_grads:
            value_of_path_to_child = loc_grad(path_value)
            grad[child] += value_of_path_to_child
            compute_gradients(child, value_of_path_to_child)

    compute_gradients(a, loss)
    return grad


tensor.__add__ = add
tensor.__sub__ = sub
tensor.__mul__ = mul
tensor.__truediv__ = div
tensor.__neg__ = neg


tensor.__floordiv__ = None
tensor.__mod__ = None
tensor.__pow__ = None

tensor.__eq__ = None
tensor.__ne__ = None
tensor.__gt__ = None
tensor.__ge__ = None
tensor.__lt__ = None
tensor.__le__ = None

tensor.__invert__ = None
tensor.__and__ = None
tensor.__or__ = None
tensor.__xor__ = None
tensor.__rshift__ = None
tensor.__lshift__ = None

tensor.__iadd__ = None
tensor.__isub__ = None
tensor.__imul__ = None
tensor.__idiv__ = None
tensor.__ifloordiv__ = None
tensor.__imod__ = None
tensor.__ipow__ = None
tensor.__irshift__ = None
tensor.__ilshift__ = None
tensor.__iand__ = None
tensor.__ixor__ = None
tensor.__ior__ = None




