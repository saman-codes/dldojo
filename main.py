# Local
from network import Network
from losses import MSE, CrossEntropy
from layers import Feedforward, Output, Convolutional
from utils import load_mnist, predict_random_mnist, get_accuracy_mnist, plot_weights, plot_random_mnist_autoencoder

# Thirdparty
import numpy as np

def run_autoencoder():
    x_train, _, _, _ = load_mnist(
        train_set_size=60000, test_set_size=0, select_label=5)
    ins = 784
    os = 784
    bs = 5000
    loss = CrossEntropy()
    net = Network()
    net.set_name('Autoencoder')
    net.add(Feedforward(shape=(250, ins)))
    net.add(Feedforward(shape=(100, 250)))
    net.add(Feedforward(shape=(250, 100)))
    net.add(Output(shape=(os, 250)))
    net.train(x_train,  x_train, loss, batch_size=bs, learning_rate=1e-3,
              epochs=100, plot_loss=True)
    plot_weights(net)
    generator = Network()
    generator.set_name('Generator')
    generator.layers = net.layers[2:]
    plot_random_mnist_autoencoder(generator)
    plot_weights()

def run_feedforward_gradient_checking():
    bs = 10
    ins = 784
    os = 10
    hs = 100
    x_train, y_train, _, _ = load_mnist(train_set_size=bs, test_set_size=0)
    ffkwargs = dict(activation='relu', use_bias=True, bias_init='zeros',
                        weight_init='glorot_uniform')
    okwargs = dict(activation='sigmoid',use_bias=True, bias_init='zeros',
                    weight_init='glorot_uniform')
    loss = CrossEntropy()
    net = Network()
    net.add(Feedforward(shape=(hs, ins), **ffkwargs))
    net.add(Feedforward(shape=(hs, hs), **ffkwargs))
    net.add(Feedforward(shape=(hs, hs), **ffkwargs))
    net.add(Output(shape=(os, hs), **okwargs))
    net.train(x_train,  y_train, loss, gradient_check=True, batch_size=bs,
                learning_rate=1, epochs=1, plot_loss=False)

def run_feedforward():
    x_train, y_train, x_test, y_test = load_mnist(train_set_size=50000, test_set_size=10000)
    ins = 784
    os = 10
    hs = 100
    bs = 1000
    loss = CrossEntropy()
    net = Network()
    net.set_name('Simple Feedforward Network')
    ffkwargs = dict(activation='relu')
    net.add(Feedforward(shape=(hs, ins), **ffkwargs))
    net.add(Feedforward(shape=(hs, hs), **ffkwargs))
    net.add(Feedforward(shape=(hs, hs), **ffkwargs))
    net.add(Output(shape=(os, hs)))
    net.train(x_train,  y_train, loss, batch_size=bs, learning_rate=1e-3,
              epochs=100, plot_loss=False)
    get_accuracy_mnist(x_test, y_test, net)
    # plot_weights(net)
    predict_random_mnist(x_test, y_test, net, save_plot=True)

def run_ff_with_regularization():
    x_train, y_train, x_test, y_test = load_mnist(train_set_size=10000, test_set_size=1000)
    ins = 784
    os = 10
    hs = 100
    bs = 1000
    loss = CrossEntropy()
    net = Network()
    net.set_name('Simple Feedforward Network')
    ffkwargs = dict(activation='relu')
    net.add(Feedforward(shape=(hs, ins), **ffkwargs))
    net.add(Feedforward(shape=(hs, hs), **ffkwargs))
    net.add(Feedforward(shape=(hs, hs), **ffkwargs))
    net.add(Feedforward(shape=(hs, hs), **ffkwargs))
    net.add(Feedforward(shape=(hs, hs), **ffkwargs))
    net.add(Output(shape=(os, hs)))
    net.train(x_train,  y_train, loss, optimizer='minibatch_sgd', batch_size=bs, learning_rate=1e-2,
            regularizer=('L2', 0.9), epochs=25, plot_loss=False)
    get_accuracy_mnist(x_test, y_test, net)

def run_ff_with_dropout():
    x_train, y_train, x_test, y_test = load_mnist(train_set_size=10000, test_set_size=1000)
    ins = 784
    os = 10
    hs = 100
    bs = 1000
    loss = CrossEntropy()
    net = Network()
    ffkwargs = dict(activation='relu', dropout=True)
    okwargs = dict(activation='sigmoid')
    net.add(Feedforward(shape=(hs, ins), **ffkwargs))
    net.add(Feedforward(shape=(hs, hs), **ffkwargs))
    net.add(Output(shape=(os, hs), **okwargs))
    net.train(x_train,  y_train, loss, optimizer='minibatch_sgd', batch_size=bs,
            learning_rate=1e-2, epochs=100, plot_loss=True)
    get_accuracy_mnist(x_test, y_test, net)

def run_ff_with_minmax_scaling():
    x_train, y_train, x_test, y_test = load_mnist(train_set_size=10000, test_set_size=1000)
    ins = 784
    os = 10
    hs = 100
    bs = 1000
    loss = CrossEntropy()
    net = Network()
    ffkwargs = dict(activation='sigmoid', weight_init='ones', preprocessing=['minmax_scaling'])
    okwargs = dict(activation='sigmoid', bias_init='zeros', weight_init='ones')
    net.add(Feedforward(shape=(hs, ins), **ffkwargs))
    net.add(Feedforward(shape=(hs, hs), **ffkwargs))
    net.add(Output(shape=(os, hs), **okwargs))
    net.train(x_train,  y_train, loss, optimizer='minibatch_sgd', batch_size=bs,
            learning_rate=1e-2, epochs=100, plot_loss=False)
    get_accuracy_mnist(x_test, y_test, net)

def run_ff_with_momentum():
    x_train, y_train, x_test, y_test = load_mnist(train_set_size=10000, test_set_size=1000)
    ins = 784
    os = 10
    hs = 100
    bs = 1000
    loss = CrossEntropy()
    net = Network()
    net.add(Feedforward(shape=(hs, ins)))
    net.add(Feedforward(shape=(hs, hs)))
    net.add(Output(shape=(os, hs)))
    net.train(x_train,  y_train, loss,
            optimizer='momentum', batch_size=bs,
            learning_rate=1e-1, epochs=100,
            plot_loss=True
            )
    get_accuracy_mnist(x_test, y_test, net)

def run_ff_with_nesterov_momentum():
    x_train, y_train, x_test, y_test = load_mnist(train_set_size=10000, test_set_size=1000)
    ins = 784
    os = 10
    hs = 100
    bs = 1000
    loss = CrossEntropy()
    net = Network()
    net.set_name('FF with nesterov momentum')
    net.add(Feedforward(shape=(hs, ins)))
    net.add(Output(shape=(os, hs)))
    net.train(x_train,  y_train, loss,
            optimizer='nesterov_momentum', batch_size=bs,
            learning_rate=1e-1, epochs=100,
            plot_loss=False
            )
    get_accuracy_mnist(x_test, y_test, net)
    plot_weights(net)

def run_no_hidden_layer_ff():
    x_train, y_train, x_test, y_test = load_mnist(train_set_size=10000, test_set_size=1000)
    ins = 784
    os = 10
    bs = 1000
    loss = CrossEntropy()
    net = Network()
    net.set_name('No hidden layer NN')
    net.add(Output(shape=(os, ins)))
    net.train(x_train,  y_train, loss,
            optimizer='nesterov_momentum', batch_size=bs,
            learning_rate=2e-1, epochs=100,
            plot_loss=False
            )
    get_accuracy_mnist(x_test, y_test, net)
    plot_weights(net)

def run_two_hidden_layers_ff_relu():
    x_train, y_train, x_test, y_test = load_mnist(train_set_size=10000, test_set_size=1000)
    ins = 784
    os = 10
    hs = 100
    bs = 1000
    loss = CrossEntropy()
    net = Network()
    net.set_name('Two hidden layers NN with Relu')
    ffkwargs = dict(activation='relu')
    net.add(Feedforward(shape=(hs, ins), **ffkwargs))
    net.add(Feedforward(shape=(hs, hs), **ffkwargs))
    net.add(Output(shape=(os, hs)))
    net.train(x_train,  y_train, loss,
            optimizer='nesterov_momentum', batch_size=bs,
            learning_rate=5e-3, epochs=100,
            plot_loss=False
            )
    get_accuracy_mnist(x_test, y_test, net)
    plot_weights(net)

def run_no_hidden_layer_ff_relu():
    x_train, y_train, x_test, y_test = load_mnist(train_set_size=10000, test_set_size=1000)
    ins = 784
    os = 10
    bs = 1000
    loss = MSE()
    net = Network()
    net.set_name('No hidden layer NN with Relu')
    net.add(Output(shape=(os, ins), activation='relu'))
    net.train(x_train,  y_train, loss,
            optimizer='nesterov_momentum', batch_size=bs,
            learning_rate=5e-2, epochs=250,
            plot_loss=True
            )
    get_accuracy_mnist(x_test, y_test, net)
    plot_weights(net)

def run_ff_with_adagrad():
    x_train, y_train, x_test, y_test = load_mnist(train_set_size=10000, test_set_size=1000)
    ins = 784
    os = 10
    hs = 100
    bs = 1000
    loss = CrossEntropy()
    net = Network()
    net.set_name('FF with Adagrad')
    net.add(Feedforward(shape=(hs, ins)))
    net.add(Output(shape=(os, hs)))
    net.train(x_train,  y_train, loss,
            optimizer='adagrad', batch_size=bs,
            learning_rate=1e-3, epochs=100,
            plot_loss=False
            )
    get_accuracy_mnist(x_test, y_test, net)
    plot_weights(net)

def run_ff_with_rmsprop():
    x_train, y_train, x_test, y_test = load_mnist(train_set_size=10000, test_set_size=1000)
    ins = 784
    os = 10
    hs = 100
    bs = 1000
    loss = CrossEntropy()
    net = Network()
    net.set_name('FF with Rmsprop')
    net.add(Feedforward(shape=(hs, ins)))
    net.add(Feedforward(shape=(hs, hs)))
    net.add(Output(shape=(os, hs)))
    net.train(x_train,  y_train, loss,
            optimizer='rmsprop', batch_size=bs,
            learning_rate=1e-4, epochs=100,
            plot_loss=True
            )
    get_accuracy_mnist(x_test, y_test, net)
    plot_weights(net)


def run_ff_with_adam():
    x_train, y_train, x_test, y_test = load_mnist(train_set_size=10000, test_set_size=1000)
    ins = 784
    os = 10
    hs = 100
    bs = 1000
    loss = CrossEntropy()
    net = Network()
    net.set_name('FF with Adam')
    net.add(Feedforward(shape=(hs, ins)))
    net.add(Output(shape=(os, hs)))
    net.train(x_train,  y_train, loss,
            optimizer='adam', batch_size=bs,
            learning_rate=1e-4, epochs=100,
            plot_loss=True, regularizer=('L2', 0.5),
            )
    get_accuracy_mnist(x_test, y_test, net)
    plot_weights(net)

def run_ff_with_softmax():
    x_train, y_train, x_test, y_test = load_mnist(train_set_size=10000, test_set_size=1000)
    ins = 784
    os = 10
    hs = 100
    bs = 1000
    loss = CrossEntropy()
    net = Network()
    net.set_name('FF with Adam')
    net.add(Feedforward(shape=(hs, ins)))
    net.add(Output(shape=(os, hs), activation='softmax'))
    net.train(x_train,  y_train, loss,
            optimizer='adam', batch_size=bs,
            learning_rate=1e-4, epochs=100,
            plot_loss=True, regularizer=('L2', 0.5),
            )
    get_accuracy_mnist(x_test, y_test, net)
    plot_weights(net)

def run_ff_with_batchnorm():
    ins = 784
    os = 10
    hs = 100
    bs = 1000
    x_train, y_train, x_test, y_test = load_mnist(train_set_size=10000, test_set_size=bs)
    loss = CrossEntropy()
    net = Network()
    net.set_name('FF with BatchNorm')
    net.add(Feedforward(shape=(hs, ins)))
    net.add(Feedforward(shape=(hs, hs), batch_normalization=True))
    net.add(Output(shape=(os, hs), activation='softmax')))
    net.train(x_train,  y_train, loss,
            optimizer='adam',
            batch_size=bs,
            learning_rate=1e-2,
            epochs=25,
            plot_loss=True,
            )

    get_accuracy_mnist(x_test, y_test, net)
    # plot_weights(net)
    # predict_random_mnist(x_test, y_test, net)

if __name__ == '__main__':
    # run_autoencoder()
    # run_feedforward()
    # run_feedforward_gradient_checking()
    # run_ff_with_dropout()
    # run_ff_with_regularization()
    # run_ff_with_minmax_scaling()
    # run_ff_with_momentum()
    # run_ff_with_nesterov_momentum()
    # run_no_hidden_layer_ff()
    # run_two_hidden_layers_ff()
    # run_two_hidden_layers_ff_relu()
    # run_no_hidden_layer_ff_relu()
    # run_ff_with_adagrad()
    # run_ff_with_rmsprop()
    # run_ff_with_adam()
    # run_ff_with_softmax()
    run_ff_with_batchnorm()






