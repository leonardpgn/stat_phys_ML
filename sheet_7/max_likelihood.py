import numpy as np
import matplotlib.pyplot as plt

input_data = []
with open("input_data.dat", "r") as input_data_file:
    for line in input_data_file:
        input_data.append(float(line))

target_data = []
with open("target_data.dat", "r") as target_data_file:
    for line in target_data_file:
        target_data.append(float(line))
target_data = np.array(target_data)

design_matrix = np.array(
    [
        len(input_data) * [1],
        input_data
    ]
).T

w = np.matmul(
        np.matmul(
            np.linalg.inv(
                np.matmul(design_matrix.T, design_matrix)
            ),
            design_matrix.T
        ),
        target_data
)


fig, ax = plt.subplots(dpi=500)
ax.scatter(input_data, target_data, marker=".", color="dodgerblue", label="data")
ax.plot(
    (-1, 1),
    (w[0] - w[1], w[0] + w[1]),
    color="orange",
    label=rf"Maximum likelihood estimation ($w_0=${w[0]:.2f}, $w_1=${w[1]:.2f})"
)
ax.set(xlabel="Input data", ylabel="Target data", xlim=(-1, 1))
ax.legend()

fig.tight_layout()
fig.savefig("./figures/max_likelihood.png")
