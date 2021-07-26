from typing import List, Union, Dict
import numpy as np
import re

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
            first_der = self.get_indefinite_integral()
            return first_der.get_nth_indefinite_integral(n - 1)
        except TypeError:
            return None

    def get_taylor_polynomial(self, n: int, x: float = 0):
        constants = {}
        for i in range(n + 1):
            d = self.get_nth_derivative(i)
            constants[i] = d.evaluate(x) / np.math.factorial(i)
        taylor_polynomial = Polynomial(constants)
        if x != 0:
            return ChainedFunction(taylor_polynomial,
                                   FunctionSubtraction(Linear(1), Constant(x)))
        return taylor_polynomial

    def get_tex_representation(self):
        return str(self)

    def roots(self, x_min: Union[float, None] = None, x_max: Union[float, None] = None) -> List[float]:
        raise NotImplementedError("root calculation not implemented")

    def x_calc(self, y: float, x_min: Union[float, None], x_max: Union[float, None]):
        raise NotImplementedError("x_calc not implemented")


class ChainedFunction(Function):
    """f(g(x))"""
    f1: Function
    f2: Function

    def evaluate(self, x: float) -> float:
        try:
            return self.f1.evaluate(self.f2.evaluate(x))
        except TypeError:
            return None

    def get_derivative(self):
        return FunctionMultiplication(ChainedFunction(
            self.f1.get_derivative(), self.f2), self.f2.get_derivative())

    def get_tex_representation(self):
        return self.f1.get_tex_representation().replace(
            "x", f"({self.f2.get_tex_representation()})")

    def __init__(self, f1: Function, f2: Function):
        self.f1 = f1
        self.f2 = f2

    def __str__(self):
        return str(self.f1).replace("x", f"({self.f2})")


class FunctionAddition(Function):
    """Two functions added together"""
    f1: Function
    f2: Function

    def evaluate(self, x: float) -> float:
        try:
            return self.f1.evaluate(x) + self.f2.evaluate(x)
        except TypeError:
            return None
    # h_method doesn't have to be overridden as slope can be derived from
    # overridden `.evaluate`

    def get_tex_representation(self):
        return f"{self.f1.get_tex_representation()} +\
     {self.f2.get_tex_representation()}"

    def get_derivative(self):
        return FunctionAddition(self.f1.get_derivative(),
                                self.f2.get_derivative())

    def get_indefinite_integral(self):
        return FunctionAddition(
            self.f1.get_indefinite_integral(),
            self.f2.get_indefinite_integral())

    def get_nth_derivative(self, n: int):
        return FunctionAddition(self.f1.get_nth_derivative(
            n), self.f2.get_nth_derivative(n))

    def get_nth_indefinite_integral(self, n: int):
        return FunctionAddition(self.f1.get_nth_indefinite_integral(n),
                                self.f2.get_nth_indefinite_integral(n))

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
        return self.__init__([f.get_indefinite_integral()
                              for f in self.functions])

    def get_tex_representation(self):
        return "(" + ' + '.join([f.get_tex_representation() for f in
                                 self.functions]) + ")"

    def __init__(self, functions: List[Function]):
        self.functions = functions

    def __str__(self):
        return f"({' + '.join([str(f) for f in self.functions])})"


class FunctionSubtraction(Function):
    """A function subtracted from another (f1 - f2)"""
    f1: Function
    f2: Function

    def evaluate(self, x: float) -> float:
        try:
            return self.f1.evaluate(x) - self.f2.evaluate(x)
        except TypeError:
            return None

    def get_tex_representation(self):
        if isinstance(self.f2, Constant):
            return f"{self.f1.get_tex_representation()} -\
                     {self.f2.get_tex_representation()}"
        return f"{self.f1.get_tex_representation()} -\
                 ({self.f2.get_tex_representation()})"

    def __init__(self, f1: Function, f2: Function):
        self.f1 = f1
        self.f2 = f2

    def __str__(self):
        if isinstance(self.f2, Constant):
            return f"{self.f1} - {self.f2}"
        return f"{self.f1} - ({self.f2})"


class FunctionMultiplication(Function):
    """Two functions multiplied with each other"""
    f1: Function
    f2: Function

    def evaluate(self, x: float) -> float:
        try:
            return self.f1.evaluate(x) * self.f2.evaluate(x)
        except TypeError:
            return None
    # h_method doesn't have to be overridden as slope can be derived from
    # overridden `.evaluate`

    def get_indefinite_integral(self):
        """S = Integral
        S(u * v) = [U * v] - S(U * v') or
        S(u * v) = [u * V] - S(u' * V)
        """
        try:
            return FunctionAddition(
                FunctionMultiplication(
                    self.f1.get_indefinite_integral(),
                    self.f2
                ),
                FunctionMultiplication(
                    Polynomial([-1]),
                    FunctionMultiplication(
                        self.f1.get_indefinite_integral(),
                        self.f2.get_derivative()
                    )
                    .get_indefinite_integral()
                )
            )
        except Exception as e:
            print(e + " (no worries for now)")
            return FunctionAddition(
                FunctionMultiplication(
                    self.f1,
                    self.f2.get_indefinite_integral()),
                FunctionMultiplication(
                    Polynomial(
                        [-1]),
                    FunctionMultiplication(
                        self.f1.get_derivative(),
                        self.f2.get_indefinite_integral()
                    )
                    .get_indefinite_integral()
                )
            )

    def get_derivative(self):
        """(u * v)' = u' * v + u * v'"""
        if issubclass(type(self.f1), Constant):
            if self.f1.constant == -1:
                if isinstance(self.f2, Sin):
                    return FunctionMultiplication(-1, Cos())
                elif isinstance(self.f2, Cos):
                    return Sin()
            elif self.f1.constant == 1:
                return self.f2.get_derivative()
        elif issubclass(type(self.f2), Constant):
            if self.f2.constant == -1:
                if isinstance(self.f1, Sin):
                    return FunctionMultiplication(-1, Cos())
                elif isinstance(self.f1, Cos):
                    return Sin()
            elif self.f2.constant == 1:
                return self.f1.get_derivative()
        else:
            return FunctionAddition(
                FunctionMultiplication(
                    self.f1.get_derivative(), self.f2), FunctionMultiplication(
                    self.f1, self.f2.get_derivative()))

            def get_tex_representation(self):
                return rf"({self.f1.get_tex_representation()}) \cdot\
                         ({self.f2.get_tex_representation})"

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


class FunctionProduct(Function):
    functions: List[Function]

    def evaluate(self, x: float) -> float:
        try:
            p = 1
            for f in self.functions:
                p *= f.evaluate(x)
            return p
        except TypeError:
            return None

    def get_derivative(self):
        def inner(l: List[Function]):
            if len(l) == 1:
                return l[0].get_derivative()
            nl = []
            for i in range(0, len(self.functions), 2):
                nl.append(FunctionMultiplication(
                    self.functions[i], self.functions[i + 1]))
            return inner(nl)
        return inner(self.functions)

    def get_tex_representation(self):
        return "(" + r" \cdot ".join([f.get_tex_representation() for f in
                                      self.functions]) + ")"

    def __init__(self, functions: List[Function]):
        self.functions = functions

    def __str__(self):
        return "(" + " * ".join([str(f) for f in self.functions]) + ")"


class Constant(Function):
    constant: float

    def evaluate(self, x: float) -> float:
        return self.constant

    def get_derivative(self):
        return self.__init__(0)

    def get_indefinite_integral(self):
        return Polynomial({1: self.constant})

    def get_tex_representation(self):
        return str(self)

    def __init__(self, constant: float):
        self.constant = constant

    def __str__(self):
        return str(self.constant)


class Linear(Function):
    constant: float

    def evaluate(self, x: float) -> float:
        return self.constant * x

    def get_derivative(self):
        return Constant(self.constant)

    def get_indefinite_integral(self):
        return Polynomial({2: self.constant / 2})

    def get_tex_representation(self):
        return str(self)

    def roots(self, x_min: Union[float, None] = None, x_max: Union[float, None] = None) -> List[float]:
        return [0.0]

    def x_calc(self, y: float, x_min: Union[float, None], x_max: Union[float, None]) -> List[float]:
        # y = cx
        # x = y/c
        return [y / self.constant]

    def __init__(self, constant: float):
        self.constant = constant

    def __str__(self):
        if self.constant == 1:
            return "x"
        elif self.constant == -1:
            return "-x"
        return f"{self.constant}x"


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

    def get_derivative(self):
        constants = self.constants.copy()
        keys = list(constants.keys())
        for i in range(len(constants)):
            tmp = constants[keys[i]]
            del constants[keys[i]]
            if not keys[i] == 0:  # f(x)' = (g(x) + c)' = g(x)'
                constants[keys[i] - 1] = tmp * keys[i]
        return Polynomial(constants)

    def get_indefinite_integral(self):
        constants = self.constants.copy()
        keys = list(constants.keys())
        ln = None
        for i in range(len(constants)):
            tmp = constants[keys[i]]
            if keys[i] == -1:
                ln = FunctionMultiplication(natural_log(), constants[keys[i]])
            del constants[keys[i]]
            constants[keys[i] + 1] = tmp * keys[i]
        if ln is None:
            return Polynomial(constants)
        return FunctionAddition(Polynomial(constants), ln)

    def get_tex_representation(self):
        s = str(self)
        n = 0
        x_s = re.finditer(r"x\^(?P<exponent>[\w\-+\.]+)", s)

        for m in x_s:
            span = m.span()
            s = s.replace(s[span[0] + n:span[1] + n],
                          "x^{" + s[span[0] + n + 2: span[1] + n] + "}")
            n += 2

        e_s = re.finditer(
            r"(?P<coefficient>-?\d\.\d+)e(?P<exponent>(\+|\-)\d+)", s)
        for m in e_s:
            groups = m.groupdict()
            c = groups["coefficient"]
            e = groups["exponent"]
            s = s.replace(f"{c}e{e}", c + r"\cdot 10^{" + e + "}")
        return s

    def roots(self, x_min: Union[float, None] = None, x_max: Union[float, None] = None) -> List[float]:
        for n in self.constants.keys():
            if n < 0:
                raise ValueError("Function includes negative exponents which \
is not supported to compute roots for.")
        coefficients = [None for _ in range(len(self.constants))]
        for n, c in self.constants.items():
            coefficients[n] = c
        return list(np.polynomial.polynomial.polyroots(np.array(coefficients)))

    def __init__(self, constants: Dict[int, float]):
        self.constants = {}
        for key in sorted(constants):
            self.constants[key] = constants[key]

    def __str__(self):
        s = ""
        first = True
        insert_minus = False
        for key, value in self.constants.items():
            if first and value < 0:
                insert_minus = True
            first = False
            if value == 0:
                continue
            operator = " + " if value > 0 else " - "
            v = str(abs(value))
            if key == 0:
                s += operator + v
                continue
            s += operator + v + "x^" + str(key)
        return ("-" if insert_minus else "") + s[3:]


class e_function(Function):
    """e^x, where x is a number"""

    def evaluate(self, x: float):
        return pow(np.e, x)

    def get_derivative(self):
        return self

    def get_tex_representation(self):
        return "e^{x}"  # necessary when called from a chained function

    def roots(self, x_min: Union[float, None] = None, x_max: Union[float, None] = None) -> List[float]:
        return []

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

    def roots(self, x_min: Union[float, None] = None, x_max: Union[float, None] = None) -> List[float]:
        return [1.0]

    def __str__(self):
        return "ln(x)"


class Sin(Function):

    def evaluate(self, x: float) -> float:
        return np.math.sin(x)

    def get_derivative(self):
        return Cos()

    def get_indefinite_integral(self):
        return FunctionMultiplication(-1, Cos())

    def roots(self, x_min: Union[float, None] = None, x_max: Union[float, None] = None) -> List[float]:
        roots = []
        if x_min is None:
            raise ValueError(
                "x_min & x_max both needed to return roots of Sin function")
        # that's called recycling, I guess
        for x in np.linspace(x_min, x_max, _standard_h):
            if x % np.pi < pow(10, -20):
                roots.append(x)
        return roots

    def __str__(self):
        return "sin(x)"


class Cos(Function):

    def evaluate(self, x: float) -> float:
        return np.math.cos(x)

    def get_derivative(self):
        return FunctionMultiplication(-1, Sin())

    def get_indefinite_integral(self):
        return Sin()

    def __str__(self):
        return "cos(x)"


class Tan(Function):

    def evaluate(self, x: float) -> float:
        return np.math.tan(x)

    def __str__(self):
        return "tan(x)"
