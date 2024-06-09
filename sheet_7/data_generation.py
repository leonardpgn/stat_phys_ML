import numpy as np
np.random.seed(42)


def f(x, a=(-.3, .5)):
    return a[0] + a[1] * x


x_values = np.random.uniform(-1, 1, 100)
fx_values = f(x_values)
t_values = fx_values + np.random.normal(0, 1/25, 100)

with open("target_data.dat", "w") as target_data_file:
    print(*t_values, file=target_data_file, sep="\n")

with open("input_data.dat", "w") as input_target_data_file:
    print(*x_values, file=input_target_data_file, sep="\n")
