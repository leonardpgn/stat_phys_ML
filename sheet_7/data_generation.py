import numpy as np


def f(x, a=(-.3, .5)):
    return a[0] + a[1] * x


x_values = np.random.uniform(-1, 1, 100)
fx_values = f(x_values)
t_values = fx_values + np.random.normal(0, 1/25, 100)

with open("data.dat", "w") as data_file:
    print(*x_values, file=data_file, sep="\n")
