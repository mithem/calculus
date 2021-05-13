from typing import List, Union
import numpy as np

_standard_h = 0.001

class Function:

    def evaluate(self, x: float) -> float:
        raise NotImplementedError()

    def h_method(self, x: float, evaluation_function = None, h: Union[None, float] = None) -> float:
        if evaluation_function is None:
            evaluation_function = self.evaluate
        if h is None:
            h = _standard_h
        hh = h / 2.0
        dy = evaluation_function(x + hh) - evaluation_function(x - hh)
        return dy / h

    def nth_derivative(self, n: int, x: float) -> float:
        if n > 2:
            return self.h_method(x, self.h_method)
        return self.h_method(x)

class Constant(Function):
    constant: float

    def evaluate(self, x: float) -> float:
        return self.constant

    def __init__(self, constant: float):
        self.constant = constant

class Linear(Function):
    constant: float
    
    def evaluate(self, x:float) -> float:
        return self.constant * x

    def __init__(self, constant: float):
        self.constant = constant

class Polynomial(Function):
    constants: [float]

    def evaluate(self, x: float) -> float:
        s = 0
        for i in range(len(self.constants)):
            s += self.constants[i] * pow(x, i)
        return s

    def __init__(self, constants: List[float]):
        self.constants = constants

    def get_derivative(self):
        constants = self.constants[1:]
        for i in range(len(constants)):
            constants[i] *= i + 1
        return Polynomial(constants)

    def get_indefinite_integral(self):
        constants = [0] + self.constants
        for i in range(1, len(constants)):
            constants[i] /= i
        return Polynomial(constants)

    def get_nth_derivative(self, n):
        constants = self.constants[n:]
        for i in range(1, len(constants)):
            constants[i] *= np.math.factorial(i)
        return Polynomial(constants)

    def get_nth_indefinite_integral(self, n):
        constants = n * [0] + self.constants
        for i in range(1, len(constants)):
            constants[i] /= np.math.factorial(i)
        return Polynomial(constants)

    def h_method(self, x: float, evaluation_function = None, h: Union[None, float] = None) -> float:
        d = self.getDerivative()
        return d.evaluate(x)
