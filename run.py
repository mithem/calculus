import matplotlib.pyplot as plt
import numpy as np
import time
from typing import List, Tuple, Union


def evaluate(functions, min_x: float,
             max_x: float = None, delta: float = None,
             points: List[Union[Tuple[float, float], float]] = []):
    t1 = time.time()
    if max_x is None:
        max_x = -min_x
    if delta is None:
        delta = 0.01

    x_points = []
    y_points = []
    for point in points:
        if type(point) == float or type(point) == np.float64:
            x = point
            y = functions[0].evaluate(x)
        else:
            x = point[0]
            y = point[1]
        x_points.append(x)
        y_points.append(y)

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

    axes.scatter(x_points, y_points)

    # for this example, you'll need to configure the matplotlib window to use
    # sensible y-axis marks
    for i in range(len(functions)):
        axes.plot(x_values, results[i], label="$%s$" %
                  functions[i].get_tex_representation())

    plt.legend()
    t2 = time.time()
    print(f"Evaluation took {t2 - t1} seconds.")
    plt.show()


def run():
    from functions import Linear, Polynomial, FunctionProduct, Sin
    print(Sin().roots(-10, 10))
    f = Polynomial({0: 2, 1: -5, 2: 1})
    roots = f.roots()
    evaluate([f], -5, points=roots)
