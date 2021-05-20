from typing import List
from functions import (Function, Constant, Linear, Polynomial, Sin, Cos, Tan,
                       FunctionSum, FunctionProduct, ChainedFunction)
import run
import argparse


def get_function_type(n: int):
    d = {
        0: Constant,
        1: Linear,
        2: Polynomial,
        3: Sin,
        4: Cos,
        5: Tan,
    }
    return d[n]


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
    if ask("Create & view more functions"):
        main()
    else:
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


def ask(question: str) -> bool:
    return input(get_context_str() + question + "? [Y/n]> ").lower().startswith("y")


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
    global functions
    contexts.append("POLY")
    pos = ask("All exponents positive")
    if pos:
        n = prompt("Highest exponent", False, expect=int)
        constants = {0: prompt("f(0)", False, expect=float)}
        for i in range(1, n + 1):
            c = prompt(f"c for n={i}", False, expect=float, optional=True)
            if c is not None:
                constants[i] = c
        f = Polynomial(constants)
    remove_from_contexts("POLY")
    return f


def not_composite_function():
    print("""0 - Constant
1 - Linear
2 - Polynomial
3 - Sin
4 - Cos
5 - Tan""")
    function = get_function_type(
        prompt("Please choose the kind of function", expect=int))
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


def menu_composite_sum():
    contexts.append("SUM")
    l = []
    n = prompt("Number of functions to add", expect=int)
    for i in range(n):
        context = f"f{i + 1}"
        contexts.append(context)
        l.append(not_composite_function())
        remove_from_contexts(context)
    remove_from_contexts("SUM")
    return FunctionSum(l)


def menu_composite_product():
    contexts.append("PROD")
    l = []
    n = prompt("Number of functions to multiply", expect=int)
    for i in range(n):
        context = f"f{i + 1}"
        contexts.append(context)
        l.append(not_composite_function())
        remove_from_contexts(context)
    remove_from_contexts("PROD")
    return FunctionProduct(l)


def menu_composite_chain():
    contexts.append("CHAIN")

    contexts.append("f1")
    f1 = not_composite_function()
    remove_from_contexts("f1")

    contexts.append("f2")
    f2 = not_composite_function()
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
    functions.append(f)


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
    composite = ask("composite function")
    if composite:
        composite_function()
    else:
        functions.append(not_composite_function())
    evaluate()


if __name__ == "__main__":
    main()
