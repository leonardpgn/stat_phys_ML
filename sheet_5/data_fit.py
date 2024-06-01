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


# c,d
def residual_error(y_true, y_pred):
    return sum((y_true - y_pred) ** 2) / len(y_true)


empirical_risk_values_train, empirical_risk_values_test = [], []
for k in range(11):
    fit_parameters, residuals = np.polyfit(train_t, train_y, k, full=True)[0:2]
    empirical_risk_values_train.append(residuals[0] / len(train_t))
    fitting_function = np.poly1d(fit_parameters)
    empirical_risk_values_test.append(residual_error(fitting_function(test_t), test_y))
print(empirical_risk_values_test)

fig, ax = plt.subplots(dpi=500)
ax.plot(range(11), empirical_risk_values_train, label="Train data")
ax.plot(range(11), empirical_risk_values_test, label="Test data")
ax.set(xlabel="Order k", ylabel="Empirical risk", xticks=range(11))
ax.legend()

fig.tight_layout()
fig.savefig("./figures/empirical_risk.png")
