import csv
import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as opt

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

# a
fig, ax = plt.subplots(dpi=500)
ax.scatter(train_t, train_y, c="blue", marker="o", label="Training data", alpha=.5)
ax.scatter(test_t, test_y, c="red", marker="o", label="Testing data", alpha=.5)
ax.set(xlabel="t", ylabel="y")
ax.legend()

fig.tight_layout()
fig.savefig("./figures/data.png")


# c
empirical_risk_values = []
for k in range(11):
    empirical_risk_values.append(np.polyfit(train_t, train_y, k, full=True)[1][0] / len(train_t))

fig, ax = plt.subplots(dpi=500)
ax.plot(range(11), empirical_risk_values)
ax.set(xlabel="Order k", ylabel="Empirical risk", xticks=range(11))

fig.tight_layout()
fig.savefig("./figures/empirical_risk.png")
