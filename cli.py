from typing import List
from functions import (Function, Constant, Linear, Polynomial, Sin, Cos, Tan,
                       FunctionAddition, FunctionMultiplication,
                       FunctionSubtraction)
import run


def get_function_type(n: int):
    d = {
        0: Constant,
        1: Linear,
        2: Polynomial,
        3: Sin,
        4: Cos,
        5: Tan,
        6: FunctionAddition,
        7: FunctionMultiplication,
        8: FunctionSubtraction
    }
    return d[n]


functions: List[Function] = []


def evaluate():
    if ask("Create & view more functions"):
        main()
    else:
        min_x = prompt("Min x", expect=float)
        max_x = prompt("Max x", expect=float, optional=True)
        run.evaluate(functions, min_x=min_x, max_x=max_x)


def ask(question: str) -> bool:
    return input(question + "? [Y/n]> ").lower().startswith("y")


def prompt(question: str, q=True, expect=str, optional=False):
    try:
        return expect(input(question +
                            ("?" if q else ":") +
                            (" [optional]" if optional else "") + " ").lower())
    except (TypeError, ValueError):
        if optional:
            return None
        return prompt(question, q, expect)


def menu_constant():
    global functions
    c = prompt("Choose a value", False, expect=float)
    f = Constant(c)
    functions.append(f)


def menu_linear():
    global function
    c = prompt("Coefficient", False, expect=float)
    f = Linear(c)
    functions.append(f)


def menu_polynomial():
    global functions
    pos = ask("All exponents positive")
    if pos:
        n = prompt("Highest exponent", False, expect=int)
        constants = {0: prompt("f(0)", False, expect=float)}
        for i in range(1, n + 1):
            c = prompt(f"c for n={i}", False, expect=float, optional=True)
            if c is not None:
                constants[n] = c
        f = Polynomial(constants)
    functions.append(f)


def main():
    global functions
    composite = ask("composite function")
    if not composite:
        print("""0 - Constant
1 - Linear
2 - Polynomial
3 - Sin
4 - Cos
5 - Tan""")
        function = get_function_type(
            prompt("Please choose the kind of function", expect=int))
        if function == Constant:
            menu_constant()
        elif function == Linear:
            menu_linear()
        elif function == Polynomial:
            menu_polynomial()
        elif function == Sin:
            functions.append(Sin())
        elif function == Cos:
            functions.append(Cos())
        elif function == Tan:
            functions.append(Tan())
        evaluate()


if __name__ == "__main__":
    main()
