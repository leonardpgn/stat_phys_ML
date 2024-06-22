import csv
import tensorflow as tf
import numpy as np
np.random.seed(42)
tf.random.set_seed(42)


class Model:
    def __init__(self, lamb, m=5):
        self.lamb = lamb
        self.m = m
        self.weights = tf.Variable(np.zeros((self.m + 1, 1)), dtype=tf.float32, name="weights")

    def __call__(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)

        poly = tf.convert_to_tensor([x ** q for q in range(self.m + 1)], dtype=tf.float32)

        if len(poly.shape) == 1:
            poly = tf.reshape(poly, (self.m + 1, 1))

        return tf.matmul(poly, self.weights, transpose_a=True)

    def loss_function(self, x, y):
        return (y - self(x))**2

    def train(self, data_t, data_y, learning_rate=0.001, epochs=1000, show_status=True):
        optimizer = tf.optimizers.Adam(learning_rate)

        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                loss = self.loss_function(data_t, data_y)
                cost = tf.reduce_sum(loss) + self.lamb * tf.reduce_sum(tf.square(self.weights))

                gradients = tape.gradient(cost, [self.weights])
                optimizer.apply_gradients(zip(gradients, [self.weights]))

            if show_status and epoch % (epochs/10) == 0:
                print(f"Epoch: {epoch}, cost: {cost.numpy():.0f}, lambda: {self.lamb:.2f}")

    def reset_weights(self):
        self.weights = tf.Variable(np.zeros((self.m + 1, 1)), dtype=tf.float32, name="weights")

    def loocv(self, data_t, data_y, training_epochs=1000, show_info=True):
        losses = []
        for i, (ti, yi) in enumerate(zip(data_t, data_y)):
            train_t = tf.concat([data_t[:i], data_t[i + 1:]], axis=0)
            train_y = tf.concat([data_y[:i], data_t[i + 1:]], axis=0)

            self.reset_weights()
            self.train(train_t, train_y, epochs=training_epochs, show_status=False)

            losses.append((self.loss_function(ti, yi)))

            if show_info:
                print(f"loocv done: {i} / {len(data_t)}, lambda: {self.lamb:.2f}")

        emp_risk = sum(losses) / len(losses)
        return emp_risk


if __name__ == "__main__":
    with open("./data/data.csv") as data_file:
        reader = csv.reader(data_file)
        data = np.array(list(reader), dtype=float)
        t_data = tf.constant(data[:, 0], dtype=tf.float32)
        y_data = tf.constant(data[:, 1], dtype=tf.float32)

    test_model = Model(lamb=0)
    print(test_model.loocv(t_data, y_data))
