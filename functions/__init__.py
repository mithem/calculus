from typing import List, Union
import numpy as np

_standard_h = 0.001

class Function:

    def evaluate(self, x: float) -> float:
        raise NotImplementedError()

    def h_method(self, x: float, h: Union[None, float] = None) -> float:
        if h is None:
            h = _standard_h
        hh = h / 2.0
        dy = self.evaluate(x + hh) - self.evaluate(x - hh)
        return dy / h


    class Addition(Function):
        f1: Function
        f2: Function

        def evaluate(self, x: float) -> float:
            return f1.evaluate(x) + f2.evaluate(x)
        # h_method doesn't have to be overridden as slope can be derived from overridden `.evaluate`

        def __init__(self, f1: Function, f2: Function):
            self.f1 = f1
            self.f2 = f2

    class Multiplication(Function):
        f1: Function
        f2: Function

        def evaluate(self, x: float) -> float:
            return f1.evaluate(x) * f2.evaluate(x)
        # h_method doesn't have to be overridden as slope can be derived from overridden `.evaluate`

        def __init__(self, f1: Function, f2: Function):
            self.f1 = f1
            self.f2 = f2

        def get_indefinite_integral(self):
            """S = Integral
            S(u * v) = [U * v] - S(U * v') or
            S(u * v) = [u * V] - S(u' * V)
            """
            try:
                return Addition(self.__init__(f1.get_indefinite_integral(), f2), self.__init__(Polynomial([-1]), self.__init__(f1.get_indefinite_integral(), f2.get_derivative()).get_indefinite_integral())
            except Exception as e:
                print(e + " (no worries for now)")
                return Addition(self.__init__(f1, f2.get_indefinite_integral()), self.__init__(Polynomial([-1]), self.__init__(f1.get_derivative(), f2.get_indefinite_integral()).get_indefinite_integral())

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

    def get_nth_derivative(self, n: int):
        constants = self.constants[n:]
        for i in range(1, len(constants)):
            constants[i] *= np.math.factorial(i)
        return Polynomial(constants)

    def get_nth_indefinite_integral(self, n: int):
        constants = n * [0] + self.constants
        for i in range(1, len(constants)):
            constants[i] /= np.math.factorial(i)
        return Polynomial(constants)

    class Addition(Function.Addition):
        f1: Polynomial
        f2: Polynomial

        def __init__(self, f1: Polynomial, f2: Polynomial):
            self.f1 = f1
            self.f2 = f2

        def get_derivative(self):
            return self.__init__(f1.get_derivative(), f2.get_derivative())

        def get_indefinite_integral(self):
            return self.__init__(f1.get_indefinite_integral(), f2.get_indefinite_integral())

        def get_nth_derivative(self, n: int):
            return self.__init__(f1.get_nth_derivative(n), f2.get_nth_derivative(n))

        def get_nth_indefinite_integral(self, n: int):
            return self.__init__(f1.get_nth_indefinite_integral(n), f2.get_nth_indefinite_integral(n))

    class Multiplication(Function.Multiplication):
        f1: Polynomial
        f2: Polynomial

        def __init__(self, f1: Polynomial, f2: Polynomial):
            self.f1 = f1
            self.f2 = f2

        def get_derivative(self):
            """(u * v)' = u' * v + u * v'"""
            return Polynomial.Addition(self.__init__(f1.get_derivative(), f2), self.__init__(f1, f2.get_derivative()))

