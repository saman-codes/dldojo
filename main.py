# Local
from network import Network
from losses import MSE, CrossEntropy
from layers import Feedforward, Output
from utils import load_mnist, predict_random_mnist
# Thirdparty
import numpy as np

'''
TODO: CONFIGURE MNIST LOADER WITHOUT KERAS, AND REMOVE KERAS AND TF AS DEPENDENCIES
'''

def main():
    x_train, y_train_bin = load_mnist()
    ins = 784
    os = 10
    hs = 30
    # loss = MSE()
    loss = CrossEntropy()
    x = x_train
    y = y_train_bin
    net = Network() 
    net.add(Feedforward(shape=(hs,ins), activation='sigmoid', use_bias=True, bias_init='zeros', weight_init='normal')) # w shape is (hs,ins)
    net.add(Feedforward(shape=(hs,hs), activation='sigmoid', use_bias=True, bias_init='zeros', weight_init='normal')) # w shape is (hs,hs)
    net.add(Output(shape=(os, hs), activation='sigmoid', use_bias=True, bias_init='zeros', weight_init='normal')) # w shape is (os, hs)
    net.train(x,  y, loss, learning_rate=5e-2, epochs=10000, verbose=False, plot_loss=True)
    predict_random_mnist(x, y, net)

if __name__=='__main__':
    main()