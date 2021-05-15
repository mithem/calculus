from typing import List, Union, Dict
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
        try:
            d = self.get_derivative()
            return d.get_nth_derivative(n - 1)
        except TypeError:
            return None

    def get_nth_indefinite_integral(self, n: int):
        if n == 0:
            return self
        try:
            return self.get_indefinite_integral().get_nth_indefinite_integral(n - 1)
        except TypeError:
            return None

    def get_taylor_polynomial(self, n: int, x: float = 0):
        constants = {}
        for i in range(n + 1):
            d = self.get_nth_derivative(i)
            constants[i] = d.evaluate(x) / np.math.factorial(i)
        return Polynomial(constants)

class FunctionAddition(Function):
    """Two functions added together"""
    f1: Function
    f2: Function

    def evaluate(self, x: float) -> float:
        try:
            return self.f1.evaluate(x) + self.f2.evaluate(x)
        except TypeError:
            return None
    # h_method doesn't have to be overridden as slope can be derived from overridden `.evaluate`

    def __init__(self, f1: Function, f2: Function):
        self.f1 = f1
        self.f2 = f2

    def __str__(self):
        return f"{self.f1} + {self.f2}"

class FunctionSum(Function):
    """Many functions added together"""
    functions: List[Function]

    def evaluate(self, x: float) -> float:
        try:
            s = 0
            for f in self.functions:
                s += f.evaluate(x)
            return s
        except TypeError:
            return None
    
    def get_derivative(self):
        return self.__init__([f.get_derivative() for f in self.functions])

    def get_indefinite_integral(self):
        return self.__init__([f.get_indefinite_integral() for f in self.functions])

    def __init__(self, functions: List[Function]):
        self.functions = functions

    def __str__(self):
        return f"({' + '.join([str(f) for f in self.functions])})"

class FunctionMultiplication(Function):
    """Two functions multiplied with each other"""
    f1: Function
    f2: Function

    def evaluate(self, x: float) -> float:
        try:
            return self.f1.evaluate(x) * self.f2.evaluate(x)
        except TypeError:
            return None
    # h_method doesn't have to be overridden as slope can be derived from overridden `.evaluate`

    def get_indefinite_integral(self):
        """S = Integral
        S(u * v) = [U * v] - S(U * v') or
        S(u * v) = [u * V] - S(u' * V)
        """
        try:
            return FunctionAddition(self.__init__(self.f1.get_indefinite_integral(), self.f2), self.__init__(Polynomial([-1]), self.__init__(self.f1.get_indefinite_integral(), self.f2.get_derivative()).get_indefinite_integral()))
        except Exception as e:
            print(e + " (no worries for now)")
            return FunctionAddition(self.__init__(self.f1, self.f2.get_indefinite_integral()), self.__init__(Polynomial([-1]), self.__init__(self.f1.get_derivative(), self.f2.get_indefinite_integral()).get_indefinite_integral()))

    def get_derivative(self):
        if issubclass(type(self.f1), Constant):
            if self.f1.constant == -1:
                if type(self.f2) == sin:
                    return FunctionMultiplication(-1, cos())
                elif type(self.f2) == cos:
                    return sin()
            elif self.f1.constant == 1:
                return self.f2.get_derivative()
        elif issubclass(type(self.f2), Constant):
            if self.f2.constant == -1:
                if type(self.f1) == sin:
                    return FunctionMultiplication(-1, cos())
                elif type(self.f1) == cos:
                    return sin()
            elif self.f2.constant == 1:
                return self.f1.get_derivative()
        else:
            return FunctionSum(FunctionMultiplication(self.f1.get_derivative(),
                    self.f2), FunctionMultiplication(self.f1,
                    self.f2.get_derivative()))

    def __init__(self, f1: Union[float, Function], f2: Union[float, Function]):
        if issubclass(type(f1), Function):
            self.f1 = f1
        else:
            self.f1 = Constant(f1)
        if issubclass(type(f2), Function):
            self.f2 = f2
        else:
            self.f2 = Constant(f2)

    def __str__(self):
        return f"({self.f1}) * ({self.f2})"

class Constant(Function):
    constant: float

    def evaluate(self, x: float) -> float:
        return self.constant

    def get_derivative(self):
        return self.__init__(0)

    def get_indefinite_integral(self):
        return Polynomial({1: self.constant})

    def __init__(self, constant: float):
        self.constant = constant

    def __str__(self):
        return str(constant)

class Linear(Function):
    constant: float
    
    def evaluate(self, x:float) -> float:
        return self.constant * x

    def __init__(self, constant: float):
        self.constant = constant

    def get_derivative(self):
        return Constant(self.constant)

    def get_indefinite_integral(self):
        return Polynomial({2: self.constant / 2})

    def __str__(self):
        return f"{constant}x"

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
    constants: Dict[int, float]

    def evaluate(self, x: float) -> float:
        try:
            s = 0
            for key, value in self.constants.items():
                s += value * x ** key
            return s
        except Exception:
            return None

    def __init__(self, constants: Dict[int, float]):
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
        ln = None
        for i in range(len(constants)):
            tmp = constants[keys[i]]
            if keys[i] == -1:
                ln =  FunctionMultiplication(natural_log(), constants[keys[i]])
            del constants[keys[i]]
            constants[keys[i] + 1] = tmp * keys[i]
        if ln is None:
            return Polynomial(constants)
        return FunctionAddition(Polynomial(constants), ln)

    def __str__(self):
        s = ""
        first = True
        insert_minus = False
        for key, value in self.constants.items():
            if value == 0:
                continue
            operator = " + " if value > 0 else " - "
            s += operator + str(abs(value)) + "x^" + str(key)
            if first and value < 0:
                insert_minus = True
            first = False
        return ("-" if insert_minus else "") + s[3:]

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

    def __str__(self):
        return f"{self.f1} + {self.f2}"  # no unnecessary parantheses

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
    """e^x, where x is a number"""

    def evaluate(self, x: float):
        return pow(np.e, x)

    def get_derivative(self):
        return self

    def __str__(self):
        return "e^x"

class natural_log(Function):
    """ln(x), where x is a number"""

    def evaluate(self, x: float):
        try:
            return np.math.log(x)
        except ValueError as e:
            if "math domain error" in str(e):
                return None

    def get_derivative(self):
        return Polynomial({-1: 1})

    def get_indefinite_integral(self, n: int):
        if n > 1:
            raise NotImplementedError()
        if n == 0:
            return self
        return self.get_derivative().get_nth_indefinite_integral(n - 1)

    def __str__(self):
        return "ln(x)"

class sin(Function):

    def evaluate(self, x: float) -> float:
        return np.math.sin(x)

    def get_derivative(self):
        return cos()

    def get_indefinite_integral(self):
        return FunctionMultiplication(-1, cos())

    def __str__(self):
        return "sin(x)"

class cos(Function):

    def evaluate(self, x: float) -> float:
        return np.math.cos(x)

    def get_derivative(self):
        return FunctionMultiplication(-1, sin())

    def get_indefinite_integral(self):
        return sin()

    def __str__(self):
        return "cos(x)"

class tan(Function):
    
    def evaluate(self, x: float) -> float:
        return np.math.tan(x)

    def __str__(self):
        return "tan(x)"
