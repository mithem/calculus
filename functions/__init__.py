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

    def get_nth_derivative(self, n: int):
        if n == 0:
            return self
        return self.get_derivative().get_nth_derivative(n - 1)

    def get_nth_indefinite_integral(self, n: int):
        if n == 0:
            return self
        return self.get_indefinite_integral().get_nth_indefinite_integral(n - 1)

class FunctionAddition(Function):
    f1: Function
    f2: Function

    def evaluate(self, x: float) -> float:
        return f1.evaluate(x) + f2.evaluate(x)
    # h_method doesn't have to be overridden as slope can be derived from overridden `.evaluate`

    def __init__(self, f1: Function, f2: Function):
        self.f1 = f1
        self.f2 = f2

class FunctionMultiplication(Function):
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
            return FunctionAddition(self.__init__(f1.get_indefinite_integral(), f2), self.__init__(Polynomial([-1]), self.__init__(f1.get_indefinite_integral(), f2.get_derivative()).get_indefinite_integral()))
        except Exception as e:
            print(e + " (no worries for now)")
            return FunctionAddition(self.__init__(f1, f2.get_indefinite_integral()), self.__init__(Polynomial([-1]), self.__init__(f1.get_derivative(), f2.get_indefinite_integral()).get_indefinite_integral()))

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
    """Examples:
    `constants = {
        -2: 1,
        -1: 3,
        0: 5,
        1: -1.5,
        2: 0.5
    }`
    => x⁻² + 3x⁻¹ + 5x⁰ -1.5x¹ + 0.5x²

    `constants = {
        -3: -2,
        0: 1,
        2: 2
    }
    `
    => -2x⁻³ + 1 + 2x²
    """
    constants: dict[int, float]

    def evaluate(self, x: float) -> float:
        try:
            s = 0
            for key, value in self.constants.items():
                s += value * x ** key
            return s
        except Exception:
            return None

    def __init__(self, constants: dict[int, float]):
        self.constants = {}
        for key in sorted(constants):
            self.constants[key] = constants[key]

    def get_derivative(self):
        constants = self.constants.copy()
        keys = list(constants.keys())
        for i in range(len(constants)):
            tmp = constants[keys[i]]
            del constants[keys[i]]
            if not keys[i] == 0: # f(x)' = (g(x) + c)' = g(x)'
                constants[keys[i] - 1] = tmp * keys[i]
        return Polynomial(constants)

    def get_indefinite_integral(self):
        constants = self.constants.copy()
        keys = list(constants.keys())
        for i in range(len(constants)):
            tmp = constants[keys[i]]
            if keys[i] == -1:
                return natural_log(constants[keys[i]])
            del constants[keys[i]]
            constants[keys[i] + 1] = tmp * keys[i]
        return Polynomial(constants)

class PolynomialAddition(FunctionAddition):
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

class PolynomialMultiplication(FunctionMultiplication):
    f1: Polynomial
    f2: Polynomial

    def __init__(self, f1: Polynomial, f2: Polynomial):
        self.f1 = f1
        self.f2 = f2

    def get_derivative(self):
        """(u * v)' = u' * v + u * v'"""
        return PolynomialAddition(self.__init__(f1.get_derivative(), f2), self.__init__(f1, f2.get_derivative()))

class e_function(Function):
    """ce^f, where c, f is a function"""
    c: Function
    f: Function

    def __init__(self, c: Union[float, Function] = 1, f: Union[float, Function] = 1):
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

class natural_log(Function):
    """c * ln(f), where c, f is a function"""
    c: Function
    f: Function

    def __init__(self, c: Union[float, Function] = 1, f: Union[float, Function] = 1):
        if issubclass(type(c), Function):
            self.c = c
        else:
            self.c = Constant(c)
        if issubclass(type(f), Function):
            self.f = f
        else:
            self.f = Linear(f)

    def evaluate(self, x: float):
        try:
            return self.c.evaluate(x) * np.math.log(self.f.evaluate(x), np.e)
        except ValueError as e:
            if "math domain error" in str(e):
                return np.nan

    def get_derivative(self):
        if issubclass(type(c), Constant):
            raise NotImplementedError()
        return FunctionAddition(FunctionMultiplication(c.get_derivative(), natural_log(1, f)), FunctionMultiplication(Polynomial({-1: 1}), f), f.get_derivative())

    def get_indefinite_integral(self, n: int):
        if n > 1:
            raise NotImplementedError()
        if n == 0:
            return self
        return self.get_derivative().get_nth_indefinite_integral(n - 1)
