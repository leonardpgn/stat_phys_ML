import csv
import numpy as np
from matplotlib import pyplot as plt

train_t, train_y = [], []
test_t, test_y = [], []

with open("./data/train.csv", "r") as train_data_file:
    reader = csv.reader(train_data_file)
    for num, row in enumerate(reader):
        if num > 0:
            train_t.append(float(row[0]))
            train_y.append(float(row[1]))

with open("./data/test.csv", "r") as test_data_file:
    reader = csv.reader(test_data_file)
    for num, row in enumerate(reader):
        if num > 0:
            test_t.append(float(row[0]))
            test_y.append(float(row[1]))

fig, ax = plt.subplots(dpi=500)
ax.scatter(train_t, train_y, c="blue", marker="o", label="Training data")
ax.scatter(test_t, test_y, c="red", marker="o", label="Testing data")
ax.set(xlabel="t", ylabel="y")
ax.legend()

fig.tight_layout()
fig.savefig("./figures/data.png")
