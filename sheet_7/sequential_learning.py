import copy
import numpy as np
from matplotlib import pyplot as plt

beta = 25

input_data = []
with open("input_data.dat", "r") as input_data_file:
    for line in input_data_file:
        input_data.append(float(line))
input_data = np.array(input_data)

target_data = []
with open("target_data.dat", "r") as target_data_file:
    for line in target_data_file:
        target_data.append(float(line))
target_data = np.array(target_data)

m = np.array([0, 0])
S = .5 * np.eye(2)
for q in range(input_data.shape[0]):
    input_batch = input_data[q:q + 10]
    target_batch = target_data[q:q + 10]
    batch_design_matrix = np.array(
        [
            list(np.ones(len(input_batch))),
            list(input_batch)
        ]
    ).T

    prev_s = copy.deepcopy(S)
    S = np.linalg.inv(
        np.linalg.inv(S) + beta * np.matmul(batch_design_matrix.T, batch_design_matrix)
    )
    m = np.matmul(
        S,
        np.matmul(
            np.linalg.inv(prev_s),
            m
        ) + beta * np.matmul(batch_design_matrix.T, target_batch)
    )

fig, ax = plt.subplots(dpi=500)
ax.scatter(input_data, target_data, marker=".", color="dodgerblue", label="data")
ax.plot(
    (-1, 1),
    (m[0] - m[1], m[0] + m[1]),
    color="orange",
    label=rf"Bayesian sequential (BS) ($w_0={m[0]:.2f}$, $w_1={m[1]:.2f}$)",
    zorder=1
)
for w_0 in (m[0] - np.sqrt(S[0, 0]), m[0] + np.sqrt(S[0, 0])):
    for w_1 in (m[1] - np.sqrt(S[1, 1]), m[1] + np.sqrt(S[1, 1])):
        ax.plot(
            (-1, 1),
            (w_0 - w_1, w_0 + w_1),
            color="gray",
            label=rf"BS ($w_0={w_0:.2f}$, $w_1={w_1:.2f}$)",
            alpha=.5,
            zorder=-1
        )
ax.set(xlim=(-1, 1))
ax.legend()

fig.tight_layout()
fig.savefig("./figures/sequential_learning.png")

