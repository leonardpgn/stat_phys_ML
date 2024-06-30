import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# load and separate data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# reshape data from (60000, 28, 28) to (60000, 28, 28, 1)
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

# normalize pixel values
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# visualize sample images
fig, axes = plt.subplots(4, 4, dpi=500)
for num, ax in enumerate(axes.flatten()):
    ax.imshow(x_train[num].reshape(28, 28), cmap='gray')
    ax.set(xticks=[], yticks=[])
fig.tight_layout()
fig.savefig('figures/sample_images.png')

# build CNN
epoch_num = 1
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=epoch_num, batch_size=128, validation_data=(x_test, y_test))
model.save(f'model/cnn_ep{epoch_num}.keras')

# evaluate test accuracy
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')
print(f'Test loss: {test_loss:.4f}')

# plot accuracy values

