import numpy as np
from functions import Function, Constant, Linear, Polynomial
from typing import Union

class natural_exponential_function(Function):

    def evaluate(x: float) -> float:
        return pow(np.e, x)

    def get_derivative():
        return self

class e_function(Function):
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

    def get_derivative(self):
        return e_function(Function.Multiplication(self, f.get_derivative()))

class natural_log_function(Function):

    def evaluate(x: float) -> float:
        return np.math.log(x, np.e)

class natural_log(Function):
    """c * ln(f), where c, f is a function"""
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
        return self.c.evaluate(x) * np.math.log(self.f.evaluate(x), base=np.e)


