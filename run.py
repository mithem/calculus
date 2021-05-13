import matplotlib.pyplot as plt
import numpy as np
import functions
from functions import Function, Linear
from functions.trig_functions import sin
from functions.e_functions import e_function

min = -0.1
max = 6
delta = 0.1

x_values = np.arange(min, max, delta)

l_f = Linear(np.e)
e_f = e_function(l_f, -1)

l = [e_f.evaluate(x) for x in x_values]

plt.plot(x_values, l)
plt.show()

