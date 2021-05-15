import matplotlib.pyplot as plt
import numpy as np
import functions
from functions import Polynomial
from functions.trig_functions import sin, cos

min = -15
max = -min
delta = 0.01

np_a = list(np.arange(min, max, delta))

x_values = []

for i in np_a:
    x_values.append(float(i))

f = Polynomial({-1: 1})

f2 = f.get_nth_indefinite_integral(1)

try:
    print(f2.constants)
except:
    pass

l1 = [f.evaluate(x) for x in x_values]
l2 = [f2.evaluate(x) for x in x_values]

x_1 = x_values.copy()
x_2 = x_values.copy()

n = 0
for i in range(len(x_1)):
    if l1[i - n] != None:
        continue
    del l1[i - n]
    del x_1[i - n]
    n += 1

n = 0
for i in range(len(x_2)):
    if l2[i - n] != None:
        continue
    del l1[i - n]
    del l2[i - n]
    del x_2[i - n]
    n += 1

plt.scatter(x_1, l1, label="f(x)") # for this example, you'll need to configure the matplotlib window to use sensible y-axis marks
plt.scatter(x_2, l2, label="f'(x)")

plt.legend()
plt.show()

