import csv
import numpy as np

with open("./data/data.csv") as data_file:
    reader = csv.reader(data_file)
    train_t = np.array(list(reader), dtype=float)[:, 0]
    train_y = np.array(list(reader), dtype=float)[:, 1]
print(train_t)


class Model:
    def __init__(self, lamb, m=5):
        self.lamb = lamb
        self.m = m

    def cost_function(self):
        return
