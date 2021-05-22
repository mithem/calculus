from typing import List, Union
from functions import (Function, Constant, Linear, Polynomial, Sin, Cos, Tan,
                       e_function, natural_log, FunctionSum, FunctionProduct,
                       ChainedFunction)
import run
import argparse


class FunctionType:
    d = {
        0: Constant,
        1: Linear,
        2: Polynomial,
        3: Sin,
        4: Cos,
        5: Tan,
        6: e_function,
        7: natural_log
    }

    @staticmethod
    def get_function_type(n: int):
        return FunctionType.d[n]

    @staticmethod
    def __str__():
        s = ""
        for n, function in FunctionType.d.items():
            s += f"{n} - {function.__name__}\n"
        return s


def get_composite_function_type(n: int):
    d = {
        0: FunctionSum,
        1: FunctionProduct,
        2: ChainedFunction
    }
    return d[n]


functions: List[Function] = []
fast_mode = False
contexts = []


class ContextNotFoundError(Exception):
    pass


def remove_from_contexts(c: str):
    for i in range(len(contexts), -1, -1):
        try:
            if contexts[i] == c:
                del contexts[i]
                return
        except IndexError:
            pass
    raise ContextNotFoundError()


def evaluate():
    if ask("Create & view more functions", default=False):
        main()
    else:
        ask_to_add_taylor_polynomial()
        min_x = prompt("Min x", expect=float)
        max_x = prompt("Max x", expect=float, optional=True, fast=False)
        delta = prompt("Evaluation delta", expect=float, optional=True,
                       fast=False)
        run.evaluate(functions, min_x=min_x, max_x=max_x, delta=delta)


def get_context_str():
    insert_context = len(contexts) > 0
    s = "[" if insert_context else ""
    if insert_context:
        for c in contexts:
            s += c + " > "
    if insert_context:
        s = s[:-3]
    s += "] " if insert_context else ""
    return s


def ask(question: str, default: Union[bool, None] = None) -> bool:
    if default is None:
        extra = "y/n"
    elif default:
        extra = "Y/n"
    else:
        extra = "y/N"
    response = input(get_context_str() + question + f"? [{extra}]> ").lower()
    if default is None:
        return response.startswith("y")
    if default:
        return not response.startswith("n")
    return response.startswith("y")


def prompt(question: str, q=True, expect=str, optional=False, fast=True):
    global fast_mode
    if fast_mode and not fast:
        return None

    try:
        return expect(input(get_context_str() + question +
                            ("?" if q else ":") +
                            (" [optional]" if optional else "") + " ").lower())
    except (TypeError, ValueError):
        if optional:
            return None
        return prompt(question, q, expect)


def ask_to_add_taylor_polynomial():
    if fast_mode:
        return
    add_taylor = ask("Add taylor polynomial of some already created function",
                     default=False)
    try:
        if add_taylor:
            i = prompt("\n".join([f"{i} - {str(functions[i])}" for i in
                                  range(len(functions))]) + "\nChoose function to add taylor \
    polynomial for", False, expect=int)
            n = prompt("Degree of the taylor polynomial", True, expect=int)
            x = prompt("x value for taylor polynomial", True, expect=float)
            f = functions[i].get_taylor_polynomial(n, x)
            functions.append(f)
            ask_to_add_taylor_polynomial()
    except Exception as e:
        print(e)
        ask_to_add_taylor_polynomial()


def menu_constant():
    global functions
    contexts.append("CONST")
    c = prompt("Choose a value", False, expect=float)
    f = Constant(c)
    remove_from_contexts("CONST")
    return f


def menu_linear():
    global functions
    contexts.append("LIN")
    c = prompt("Coefficient", False, expect=float)
    f = Linear(c)
    remove_from_contexts("LIN")
    return f


def menu_polynomial():
    def positive_ns():
        contexts.append("POS-EXP")
        n = prompt("Highest exponent", False, expect=int)
        for i in range(1, n + 1):
            c = prompt(f"c for n={i}", False, expect=float, optional=True)
            if c is not None:
                constants[i] = c
        remove_from_contexts("POS-EXP")

    def negative_ns():
        contexts.append("NEG-EXP")
        n = prompt("Number of exponents below 0", False, expect=int)
        for i in range(-n, 0):
            c = prompt(f"c for n={i}", False, expect=float, optional=True)
            if c is not None:
                constants[i] = c
        remove_from_contexts("NEG-EXP")

    global functions
    contexts.append("POLY")
    constants = {}
    c0 = prompt("f(0)", False, expect=float, optional=True)
    if c0 is not None:
        constants[0] = c0
    neg = ask("Are there negative exponents", default=False)
    if neg:
        negative_ns()
    positive_ns()
    f = Polynomial(constants)
    remove_from_contexts("POLY")
    return f


def not_composite_function():
    function = FunctionType.get_function_type(
        prompt(f"{FunctionType.__str__()}Please choose the kind of function",
               expect=int))
    if function == Constant:
        return menu_constant()
    elif function == Linear:
        return menu_linear()
    elif function == Polynomial:
        return menu_polynomial()
    elif function == Sin:
        return Sin()
    elif function == Cos:
        return Cos()
    elif function == Tan:
        return Tan()
    elif function == e_function:
        return e_function()
    elif function == natural_log:
        return natural_log()


def menu_composite_sum():
    contexts.append("SUM")
    functions = []
    n = prompt("Number of functions to add", expect=int)
    for i in range(n):
        context = f"f{i + 1}"
        contexts.append(context)
        functions.append(not_composite_function())
        remove_from_contexts(context)
    remove_from_contexts("SUM")
    return FunctionSum(functions)


def menu_composite_product():
    contexts.append("PROD")
    functions = []
    n = prompt("Number of functions to multiply", expect=int)
    for i in range(n):
        context = f"f{i + 1}"
        contexts.append(context)
        functions.append(not_composite_function())
        remove_from_contexts(context)
    remove_from_contexts("PROD")
    return FunctionProduct(functions)


def menu_composite_chain():
    contexts.append("CHAIN")

    contexts.append("f1")
    f1 = choose_function()
    remove_from_contexts("f1")

    contexts.append("f2")
    f2 = choose_function()
    remove_from_contexts("f2")

    remove_from_contexts("CHAIN")
    return ChainedFunction(f1, f2)


def composite_function():
    function = get_composite_function_type(prompt("0 - Sum\n1 - Product\n2 - \
Chain\nPlease choose the kind of function", expect=int))
    if function == FunctionSum:
        f = menu_composite_sum()
    elif function == FunctionProduct:
        f = menu_composite_product()
    elif function == ChainedFunction:
        f = menu_composite_chain()
    return f


def choose_function():
    composite = ask("composite function", default=False)
    if composite:
        return composite_function()
    else:
        return not_composite_function()


def main():
    global functions, fast_mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", "-f", action="store_true",
                        help="Fast mode. Skip optional parameters")
    parser.add_argument("--run", "-r", action="store_true",
                        help="Execute run.py:run() and exit.")
    args = parser.parse_args()
    if args.run:
        run.run()
        exit(0)
    fast_mode = args.fast
    functions.append(choose_function())
    evaluate()


if __name__ == "__main__":
    main()
