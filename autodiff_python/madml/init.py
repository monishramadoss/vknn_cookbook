import functools
from madml.tensor import tensor


def zeros(s):
    return tensor([0 for _ in range(functools.reduce(lambda x, y: x * y, s, 1))], s)
