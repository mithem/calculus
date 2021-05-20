import matplotlib.pyplot as plt
import numpy as np
import time


def evaluate(functions, min_x: float, max_x: float = None,
             delta: float = None):
    t1 = time.time()
    if max_x is None:
        max_x = -min_x
    if delta is None:
        delta = 0.01

    np_a = list(np.arange(min_x, max_x, delta))
    x_values = list(map(lambda x: float(x), np_a))

    # does that have lower memory consumption than just x_values?
    results = [[] for i in range(len(functions))]

    for i in range(len(functions)):
        results[i] = [functions[i].evaluate(x) for x in x_values]

    min_y = None
    max_y = None
    final_x_values = [[] for i in range(len(x_values))]
    for i in range(len(results)):
        for j in range(len(results[i])):
            v = results[i][j]
            if v is not None:
                # second comparison isn't evaluated if min_y == None
                if min_y is None or v < min_y:
                    min_y = v
                if max_y is None or v > max_y:
                    max_y = v
                final_x_values[i].append(j)

    lim_min_y = min_y - np.abs(min_y) * 0.15
    lim_max_y = max_y + np.abs(max_y) * 0.15

    axes = plt.gca()
    axes.set_ylim([lim_min_y, lim_max_y])

    # for this example, you'll need to configure the matplotlib window to use
    # sensible y-axis marks
    for i in range(len(functions)):
        plt.plot(x_values, results[i], label="$%s$" %
                 functions[i].get_tex_representation())

    plt.legend()
    t2 = time.time()
    print(f"Evaluation took {t2 - t1} seconds.")
    plt.show()


def run():
    from functions import Linear, Polynomial, FunctionProduct
    f = Linear(1)
    g = FunctionProduct([f, f, f])
    h = Polynomial({3: 1.5})
    evaluate([g, h], -5)
