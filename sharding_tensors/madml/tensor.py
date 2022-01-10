from madml.manager import graph_mgr
from functools import reduce
from collections import defaultdict


# todo  introduce context managers

class tensor(object):
    def __init__(self, data, shape=None, dtype='float32', local_grads=[]):
        if shape is None:
            shape = []
        self.data = data
        self.shape = shape
        self.dtype = dtype
        self.local_grads = local_grads

        self.input_layer_name = ''
        self.layer = ''

        self.ndim = len(shape)
        self._stride = [1]
        self._stride += [shape[i - 1] * self._stride[i - 1] for i in range(self.ndim)]
        self.coherency_map = {}
        self._size = [reduce(lambda x, y: x * y, self.shape[i:]) for i in range(self.ndim)]

    def __len__(self):
        return sum(self.shape)

    def __str__(self):
        return self.input_layer_name + ' ' + str(self.shape) + ' ' + 'layer_name: ' + self.layer

    def view_idx(self, axis=-1):
        return self._stride[axis]

    def backward(self):
        return get_gradient(self, self.local_grads[0])

    def size(self, idx=0):
        return self._size[idx]

    def shard(self, broad_cast: bool = False):
        if broad_cast:
            pass
        pass

    def scatter(self, shard_shape=[]):
        pass

    def gather(self, *args, **kwargs):
        pass


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
        (a, lambda path_value: path_value.reshape(a.shape))
    ]
    return tensor([0 for _ in range(len(a))], shape=shape, dtype=a.dtype, local_grads=local_grads)


def transpose(a, axes):
    local_grads = [
        (a, lambda path_value: transpose(path_value, axes))
    ]
    shape = [a.shape[axes[i]] for i in range(len(axes))]
    return tensor([0 for _ in range(len(a))], shape=shape, dtype=a.dtype, local_grads=local_grads)


def flatten(a):
    local_grads = [
        (a, lambda path_value: path_value.reshape(a.shape))
    ]
    shape = [a.shape[0], a.size(1)]
    return tensor([0 for _ in range(len(a))], shape=shape, dtype=a.dtype, local_grads=local_grads)


def unfold(a, kernel, stride, padding, dilation):
    n, c = a.shape[0], a.shape[1]
    _shape = a.shape[2:]
    ten = _shape.copy()
    col = [0 for _ in _shape]

    for i, s in enumerate(_shape):
        col[i] = s + 2 * padding[i] - dilation[i] * (kernel[i] - 1) - 1
        col[i] = col[i] / stride[i]

    output_length = 1
    for i in range(len(col)):
        output_length *= col[i]

    local_grads = [
        (a, lambda path_value: fold(path_value, col, ten, kernel, stride, padding, dilation))
    ]

    return tensor([0 for _ in range(n * c * len(kernel) * output_length)], shape=[n, c * sum(kernel), output_length],
                  dtype=a.dtype, local_grads=local_grads)


def fold(a, ten, kernel, stride, padding, dilation):
    local_grads = [
        (a, lambda path_value: unfold(path_value, kernel, stride, padding, dilation))
    ]
    n, c = a.shape[0], a.shape[1] / sum(kernel),
    ten_size = 1
    for t in ten:
        ten_size *= t
    return tensor([0 for _ in range(n*c*ten_size)], shape=[n, c, ten], dtype=a.dtype, local_grads=local_grads)


def matmul(a, b):
    local_grads = [
        (a, lambda path_value: a),
        (b, lambda path_value: b),
    ]
    return tensor([0 for _ in range(a.shape[0] * b.shape[1])], shape=[a.shape[0], b.shape[1]],
                  dtype=a.dtype, local_grads=local_grads)


def concatenate(a, b, axis=-1):
    from operator import add, mul
    local_grads = [
        (a, lambda path_value: b),
        (b, lambda path_value: a),
    ]

    shape_a = a.shape
    shape_b = b.shape
    shape_c = shape_a

    if a.ndim == b.ndim:
        shape_c[axis] += shape_b[axis]
    else:
        pass

    new_size = sum(shape_c)

    return tensor([0 for _ in range(new_size)], shape=shape_c, dtype=a.dtype, local_grads=local_grads)


def get_gradient(a: tensor, loss: tensor):
    grad = defaultdict(lambda: tensor(0, [1]))

    def compute_gradients(var: tensor, path_value: tensor):
        for child, loc_grad in var.local_grads:
            value_of_path_to_child = loc_grad(path_value)
            grad[child] += value_of_path_to_child
            compute_gradients(child, value_of_path_to_child)

    compute_gradients(a, loss)
    return grad


tensor.reshape = reshape
tensor.transpose = transpose
tensor.flatten = flatten
tensor.fold = fold
tensor.unfold = unfold

tensor.__add__ = add
tensor.__sub__ = sub
tensor.__mul__ = mul
tensor.__truediv__ = div
tensor.__floordiv__ = None
tensor.__mod__ = None
tensor.__pow__ = None
tensor.__rshift__ = None
tensor.__lshift__ = None
tensor.__and__ = None
tensor.__or__ = None
tensor.__xor__ = None

tensor.__lt__ = None
tensor.__gt__ = None
tensor.__ge__ = None
tensor.__le__ = None
tensor.__eq__ = None
tensor.__ne__ = None

tensor.__neg__ = neg
tensor.__pos__ = None
tensor.__invert__ = None

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
tensor.__ior__ = None
tensor.__ixor__ = None
