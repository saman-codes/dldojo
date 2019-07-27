# Standard Python
import random
# Thirdparty
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

def load_mnist():
    n_classes = 10
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train[0:100]
    y_train = y_train[0:100]
    x_train = np.moveaxis(x_train, 0, -1)
    x_train = x_train.reshape(28*28, 100)
    y_train_bin = np.eye(n_classes)[y_train].T
    return x_train, y_train_bin

def predict_random_mnist(x, y, net):
    for i in range(10):
        s = x.shape[1]
        i = random.randint(0,s-1)
        img = x[:,i].reshape(28, 28)
        pred = np.argmax(net.predict(x[:,i]))
        plt.imshow(img)
        plt.title(f'Predicted: {pred}')
        plt.show()