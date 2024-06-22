import csv
from matplotlib import pyplot as plt
from model import Model
import numpy as np
import tensorflow as tf

with open("./data/data.csv") as data_file:
    reader = csv.reader(data_file)
    data = np.array(list(reader), dtype=float)
    t_data = tf.constant(data[:, 0], dtype=tf.float32)
    y_data = tf.constant(data[:, 1], dtype=tf.float32)

lamb_values = np.linspace(0, 5, 100)
empirical_risk = []

for num, lamb in enumerate(lamb_values):
    print(f"run {num} / {len(lamb_values)}")
    model = Model(lamb=lamb)
    empirical_risk.append(model.loocv(t_data, y_data, training_epochs=10))

empirical_risk = np.array(empirical_risk)

fig, ax = plt.subplots(dpi=500)
ax.plot(lamb_values, empirical_risk, label="empirical risk")
ax.set(xlabel="lambda", ylabel="empirical risk")
ax.legend()

fig.tight_layout()
fig.savefig("./figures/hyperparameter_search.png")
