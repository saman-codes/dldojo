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


def load_mnist(train_set_size=1000, test_set_size=100, select_label=None):
    x_train, y_train = mnist.train_images(), mnist.train_labels()
    x_test, y_test = mnist.test_images(), mnist.test_labels()
    if isinstance(select_label, int):
        x_train = x_train[np.where(y_train == select_label)]
        x_test = x_test[np.where(y_test == select_label)]
        y_train = y_train[np.where(y_train == select_label)]
        y_test = y_test[np.where(y_test == select_label)]
        train_set_size = y_train.size
        test_set_size = y_test.size
    x_train = x_train[0:train_set_size] / 255
    y_train = y_train[0:train_set_size]
    x_test = x_test[0:test_set_size] / 255
    y_test = y_test[0:test_set_size]
    x_train = np.moveaxis(x_train, 0, -1)
    x_train = x_train.reshape(28 * 28, train_set_size)
    x_test = np.moveaxis(x_test, 0, -1)
    x_test = x_test.reshape(28 * 28, test_set_size)
    y_train_bin = np.eye(10)[y_train].T
    y_test_bin = np.eye(10)[y_test].T
    return x_train, y_train_bin, x_test, y_test_bin


def plot_random_mnist_autoencoder(net, save_plot=False, show_plot=True):
    cols, rows = 3, 3
    fig = plt.figure(figsize=(5, 5))
    plt.title(f'{net.__name__}', fontsize=12, y=1.08)
    plt.xticks([])
    plt.yticks([])
    input_layer = net.layers[0]
    input_size = input_layer.shape[1]
    for idx in range(1, cols * rows + 1):
        x = np.random.rand(input_size, 1)
        pred = net.test_predict(x)
        img = pred.reshape((28, 28))
        fig.add_subplot(rows, cols, idx)
        plt.gray()
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    if save_plot:
        net_name = '_'.join(net.__name__.split(' '))
        plt.savefig(os.path.join(settings.IMG_DIR, f'{net_name}.jpg'))
    if show_plot:
        plt.show()


def predict_random_mnist(x, y, net, save_plot=False, show_plot=True):
    cols, rows = 3, 3
    fig = plt.figure(figsize=(5, 5))
    plt.title(f'{net.__name__}', fontsize=12, y=1.08)
    plt.xticks([])
    plt.yticks([])
    for idx in range(1, cols * rows + 1):
        s = x.shape[1]
        i = random.randint(0, s - 1)
        img = x[:, i].reshape(28, 28)
        pred = np.argmax(net.test_predict(x[:, i].reshape(-1, 1)))
        fig.add_subplot(rows, cols, idx)
        plt.gray()
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'Predicted: {pred}', fontsize=10)
    if save_plot:
        net_name = '_'.join(net.__name__.split(' '))
        plt.savefig(os.path.join(settings.IMG_DIR, f'{net_name}.jpg'))
    if show_plot:
        plt.show()


def get_accuracy_mnist(x, y, net):
    y_pred = net.test_predict(x)
    y_pred_scalar = np.argmax(y_pred, axis=0)
    y_scalar = np.argmax(y, axis=0)
    accuracy = np.sum(y_scalar == y_pred_scalar) / len(y_scalar)
    logging.info(f'{net.__name__} - Accuracy: {100*accuracy}%')


def plot_weights(net, save_plot=False, show_plot=True):
    for idx, layer in enumerate(net.layers):
        cols = int(np.sqrt(layer.shape[0]))
        rows = int(layer.shape[0] / cols)
        if layer.shape[0] % rows > 0:
            rows += 1
        fig = plt.figure(figsize=(5, 5))
        plt.title(f'{net.__name__} weights - Layer {idx}', fontsize=12)
        plt.xticks([])
        plt.yticks([])
        for neuron in range(layer.shape[0]):
            fig.add_subplot(rows, cols, neuron + 1)
            square_shape = (int(np.sqrt(layer.weights.shape[1])), -1)
            w = layer.weights[neuron, :].reshape(square_shape)
            plt.imshow(w, cmap='gray')
            plt.xticks([])
            plt.yticks([])

        if save_plot:
            net_name = '_'.join(net.__name__.split(' '))
            plt.savefig(
                os.path.join(settings.IMG_DIR,
                             f'{net_name}_layer_{idx}_weights.jpg'))
        if show_plot:
            plt.show()
