# Local
from network import Network
from losses import MSE, CrossEntropy
from layers import Feedforward, Output, Convolutional
from utils import load_mnist, predict_random_mnist, get_accuracy_mnist, plot_weights, plot_random_mnist_autoencoder
# Thirdparty
import numpy as np

def run_autoencoder():
    x_train, _, _, _ = load_mnist(
        train_set_size=60000, test_set_size=0, select_label=0)
    ins = 784
    os = 784
    bs = 5000
    # loss = MSE()
    loss = CrossEntropy()
    net = Network()
    net.set_name('Autoencoder')
    net.add(Feedforward(shape=(250, ins), activation='sigmoid',
                        use_bias=True, bias_init='zeros', weight_init='normal'))
    net.add(Feedforward(shape=(100, 250), activation='sigmoid',
                        use_bias=True, bias_init='zeros', weight_init='normal'))
    net.add(Feedforward(shape=(250, 100), activation='sigmoid',
                        use_bias=True, bias_init='zeros', weight_init='normal'))
    net.add(Output(shape=(os, 250), activation='sigmoid',
                   use_bias=True, bias_init='zeros', weight_init='normal'))
    net.train(x_train,  x_train, loss, batch_size=bs, learning_rate=5,
              epochs=25, verbose=False, plot_loss=True)
    plot_weights(net)
    generator = Network()
    generator.set_name('Generator')
    generator.layers = net.layers[2:]
    plot_random_mnist_autoencoder(generator)

def run_feedforward_gradient_checking():
    bs = 1
    ins = 784
    os = 10
    hs = 10
    x_train, y_train, _, _ = load_mnist(train_set_size=bs, test_set_size=0)
    # x_train = np.random.random((ins, bs))
    # y_train = np.ones((ins, bs))*10
    # loss = MSE()
    loss = CrossEntropy()
    net = Network()
    net.add(Feedforward(shape=(hs, ins), activation='relu',
                        use_bias=True, bias_init='normal', weight_init='normal'))
    # net.add(Feedforward(shape=(hs, hs), activation='relu',
    #                     use_bias=True, bias_init='normal', weight_init='normal'))
    net.add(Output(shape=(os, hs), activation='sigmoid',
                   use_bias=True, bias_init='normal', weight_init='normal'))
    net.train(x_train,  y_train, loss, gradient_check=True, batch_size=bs, learning_rate=1,
              epochs=1, plot_loss=False)

def run_feedforward():
    x_train, y_train, x_test, y_test = load_mnist(train_set_size=10000, test_set_size=1000)
    ins = 784
    os = 10
    hs = 100
    bs = 1000
    # loss = MSE()
    loss = CrossEntropy()
    net = Network()
    net.set_name('Simple Feedforward Network')
    net.add(Feedforward(shape=(hs, ins), activation='sigmoid',
                        use_bias=True, bias_init='normal', weight_init='normal',
                        minmax_scaling=False))
    net.add(Feedforward(shape=(hs, hs), activation='sigmoid',
                        use_bias=True, bias_init='normal', weight_init='normal',
                        minmax_scaling=False))
    net.add(Output(shape=(os, hs), activation='sigmoid',
                   use_bias=True, bias_init='normal', weight_init='normal',
                   minmax_scaling=False))
    net.train(x_train,  y_train, loss, batch_size=bs, learning_rate=1e-3,
              epochs=100, regularizer=('L2', 0.3), verbose=False, plot_loss=False)
    get_accuracy_mnist(x_test, y_test, net)
    # plot_weights(net)
    # predict_random_mnist(x_test, y_test, net, save_plot=True)

def run_cnn():
    x_train, y_train, x_test, y_test = load_mnist(
        train_set_size=10000, test_set_size=10000)
    ins = (28, 28)
    os = 10
    hs = 100
    bs = 1000
    # loss = MSE()
    loss = CrossEntropy()
    net = Network()
    net.set_name('Simple Feedforward Network')
    net.add(Convolutional(shape=(hs, *ins), activation='sigmoid',
                        use_bias=True, bias_init='zeros', weight_init='normal'))
    net.add(Convolutional(shape=(hs, *ins), activation='sigmoid',
                        use_bias=True, bias_init='zeros', weight_init='normal'))
    net.add(Feedforward(shape=(hs, hs), flatten=True, activation='sigmoid',
                        use_bias=True, bias_init='zeros', weight_init='normal'))
    net.add(Output(shape=(os, hs), activation='softmax',
                   use_bias=True, bias_init='zeros', weight_init='normal'))
    net.train(x_train,  y_train, loss, gradient_check=True, batch_size=bs, learning_rate=1,
              epochs=100, regularizer=('L2', 0.3), verbose=False, plot_loss=False)
    get_accuracy_mnist(x_test, y_test, net)

if __name__ == '__main__':
    # run_autoencoder()
    run_feedforward()
    # run_feedforward_gradient_checking()
    # run_cnn()

