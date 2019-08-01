# Standard Python
import os
import random
import logging

# Local
import settings

# Thirdparty
import mnist
import numpy as np
import matplotlib.pyplot as plt


def load_mnist(train_set_size=1000, test_set_size=100):
    x_train, y_train = mnist.train_images(), mnist.train_labels()
    x_test, y_test = mnist.test_images(), mnist.test_labels()
    x_train = x_train[0:train_set_size]
    y_train = y_train[0:train_set_size]
    x_test = x_test[0:test_set_size]
    y_test = y_test[0:test_set_size]
    x_train = np.moveaxis(x_train, 0, -1)
    x_train = x_train.reshape(28*28, train_set_size)
    x_test = np.moveaxis(x_test, 0, -1)
    x_test = x_test.reshape(28*28, test_set_size)
    y_train_bin = np.eye(10)[y_train].T
    y_test_bin = np.eye(10)[y_test].T
    return x_train, y_train_bin, x_test, y_test_bin

def plot_random_mnist_autoencoder(net, save_plot=False):
    cols = 3
    rows = 3
    fig = plt.figure(figsize=(5,5))
    plt.title(f'{net.__name__}', fontsize=12, y=1.08)
    plt.xticks([])
    plt.yticks([])
    for idx in range(1, cols*rows+1):
        x = np.random.rand(784, 1)
        pred = net.predict(x)
        img = pred.reshape((28, 28))
        fig.add_subplot(rows, cols, idx)
        plt.gray()
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    if save_plot:
        plt.savefig(os.path.join(settings.FIGURES_DIR, f'{net.__name__}.png'))
    plt.show()

def predict_random_mnist(x, y, net, save_plot=False):
    cols = 3
    rows = 3
    fig = plt.figure(figsize=(5,5))
    plt.title(f'{net.__name__}', fontsize=12, y=1.08)
    plt.xticks([])
    plt.yticks([])
    for idx in range(1, cols*rows+1):
        s = x.shape[1]
        i = random.randint(0, s-1)
        img = x[:, i].reshape(28, 28)
        pred = np.argmax(net.predict(x[:, i]))
        fig.add_subplot(rows, cols, idx)
        plt.gray()
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Predicted: {pred}', fontsize=10)
    if save_plot:
        plt.savefig(os.path.join(settings.FIGURES_DIR, f'{net.__name__}.png'))
    plt.show()

def get_accuracy_mnist(x, y, net):
    y_pred = net.predict(x)
    y_pred_scalar = np.argmax(y_pred, axis=0)
    y_scalar = np.argmax(y, axis=0)
    accuracy = np.sum(y_scalar == y_pred_scalar)/len(y_scalar)
    logging.info(f'Accuracy: {100*accuracy}%')

def plot_weights(net, save_plot=True):
    cols = len(net.layers)
    rows = 1
    fig = plt.figure(figsize=(5,5))
    plt.title(f'{net.__name__} weights', fontsize=12)
    plt.xticks([])
    plt.yticks([])
    for idx, layer in enumerate(net.layers):
        w = layer.weights
        fig.add_subplot(rows, cols, idx+1)
        plt.imshow(w)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Layer {idx} weights', fontsize=10)
    if save_plot:
        plt.savefig(os.path.join(settings.FIGURES_DIR, f'{net.__name__}_weights.png'))
    plt.show()
