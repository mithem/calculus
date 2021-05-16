import matplotlib.pyplot as plt
import numpy as np
import sys
import time
from functions import cos


def evaluate(f1, f2, min_x: float, max_x: float = None, delta: float = None):
    if max_x is None:
        max_x = -min_x
    if delta is None:
        delta = 0.01

    np_a = list(np.arange(min_x, max_x, delta))
    x_values = []
    for i in np_a:
        x_values.append(float(i))

    l1 = [f1.evaluate(x) for x in x_values]
    l2 = [f2.evaluate(x) for x in x_values]

    x_1 = x_values.copy()
    x_2 = x_values.copy()

    n = 0
    for i in range(len(x_1)):
        if l1[i - n] is not None:
            continue
        del l1[i - n]
        del x_1[i - n]
        n += 1

    n = 0
    for i in range(len(x_2)):
        if l2[i - n] is not None:
            continue
        del l2[i - n]
        del x_2[i - n]
        n += 1

    min_y = np.min(l1)
    max_y = np.max(l1)

    lim_min_y = min_y - np.abs(min_y) * 0.15
    lim_max_y = max_y + np.abs(max_y) * 0.15

    axes = plt.gca()
    axes.set_ylim([lim_min_y, lim_max_y])

    # for this example, you'll need to configure the matplotlib window to use
    # sensible y-axis marks
    plt.plot(x_1, l1, label="$%s$"%f1.get_tex_representation())
    plt.plot(x_2, l2, label="$%s$"%f2.get_tex_representation())

    plt.legend()
    plt.show()


try:
    n = int(sys.argv[1])
    min_x = float(sys.argv[2])
except IndexError:
    print("Expected 'python3 run.py n min_x' where n is the amount of derivatives (incl. 0) used to compute the tailor polynomial and min_x the minimum x value computed (max is symmetrical to y-axis)")
    exit(-1)

f1 = cos()

t1 = time.time()
f2 = f1.get_taylor_polynomial(n)
t2 = time.time()

print(f"{t2 - t1} seconds")
print(f"Desired taylor polynomial: {f2}")

evaluate(f1, f2, min_x)
