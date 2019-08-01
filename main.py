# Local
from network import Network
from losses import MSE, CrossEntropy
from layers import Feedforward, Output
from utils import load_mnist, predict_random_mnist, get_accuracy_mnist, plot_weights, plot_random_mnist_autoencoder
# Thirdparty
import numpy as np

def run_autoencoder():
    x_train, _, _, _ = load_mnist(
        train_set_size=10000, test_set_size=0)
    ins = 784
    os = 784
    hs = 100
    bs = 100
    # loss = MSE()
    loss = CrossEntropy()
    net = Network()
    net.set_name('Autoencoder')
    net.add(Feedforward(shape=(hs, ins), activation='sigmoid',
                        use_bias=True, bias_init='zeros', weight_init='normal'))
    net.add(Feedforward(shape=(hs, hs), activation='sigmoid',
                        use_bias=True, bias_init='zeros', weight_init='normal'))
    net.add(Feedforward(shape=(hs, hs), activation='sigmoid',
                        use_bias=True, bias_init='zeros', weight_init='normal'))
    net.add(Output(shape=(os, hs), activation='sigmoid',
                   use_bias=True, bias_init='zeros', weight_init='normal'))
    net.train(x_train,  x_train, loss, batch_size=bs, learning_rate=5,
              epochs=100, verbose=False, plot_loss=True)
    
    # get_accuracy_mnist(x_test, x_test, net)
    # plot_weights(net)
    plot_random_mnist_autoencoder(net)


def run_feedforward():
    x_train, y_train, x_test, y_test = load_mnist(
        train_set_size=1000, test_set_size=100)
    ins = 784
    os = 10
    hs = 100
    bs = 1000
    # loss = MSE()
    loss = CrossEntropy()
    net = Network()
    net.set_name('SimpleFeedforward')
    net.add(Feedforward(shape=(hs, ins), activation='sigmoid',
                        use_bias=True, bias_init='zeros', weight_init='normal'))
    net.add(Feedforward(shape=(hs, hs), activation='sigmoid',
                        use_bias=True, bias_init='zeros', weight_init='normal'))
    net.add(Output(shape=(os, hs), activation='sigmoid',
                   use_bias=True, bias_init='zeros', weight_init='normal'))
    net.train(x_train,  y_train, loss, batch_size=bs, learning_rate=5,
              epochs=100, verbose=False, plot_loss=True)
    get_accuracy_mnist(x_test, y_test, net)
    plot_weights(net)
    predict_random_mnist(x_test, y_test, net, save_plot=True)


if __name__ == '__main__':
    run_autoencoder()
