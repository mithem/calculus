import matplotlib.pyplot as plt
import numpy as np
import functions
from functions import Polynomial
from functions.trig_functions import sin, cos
from functions.e_functions import e_function

min = -15
max = -min
delta = 0.1

x_values = np.arange(min, max, delta)

f = Polynomial([1, 0, -0.5, 0, 1/24, 0, -1/np.math.factorial(6), 0, 1/np.math.factorial(8)])
f2 = Polynomial([1])

function = f2
F = f2.get_nth_indefinite_integral(1)

l1 = [f2.evaluate(x) for x in x_values]
l5 = [F.evaluate(x) for x in x_values]

plt.plot(x_values, l1, label="f(x)")
plt.plot(x_values, l5, label="F(x)")

plt.legend()
plt.show()

