import numpy as np
from functions import Function, Constant, Linear
from typing import Union

class natural_exponential_function:

    def evaluate(x: float) -> float:
        return pow(np.e, x)

    def makeDerivative():
        return self

class e_function:
    """ce^f, where c, f is a function"""
    c: Function
    f: Function

    def __init__(self, c: Union[float, Function], f: Union[float, Function]):
        if issubclass(type(c), Function):
            self.c = c
        else:
            self.c = Constant(c)
        if issubclass(type(f), Function):
            self.f = f
        else:
            self.f = Linear(f)

    def evaluate(self, x: float):
        return self.c.evaluate(x) * pow(np.e, self.f.evaluate(x))
