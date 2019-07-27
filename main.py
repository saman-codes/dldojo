# Local
from network import Network
from losses import MSE, CrossEntropy
from layers import Feedforward, Output
# Thirdparty
import numpy as np

def main():
    # bs = 7
    # ins = 784
    # os = 10
    # hs = 5
    bs = 1
    ins = 2
    os = 4
    hs = 3
    x = np.random.uniform(-1,1, size=(ins, bs))
    y = np.zeros((os, bs))
    # y = np.array([[1], [0], [1], [0]])
    loss = MSE()
    # loss = CrossEntropy()
    # x = x_train
    # y = y_train_bin
    net = Network() 
    net.add(Feedforward(shape=(hs,ins), activation='sigmoid', use_bias=True, weight_init='normal')) # w shape is (hs,ins)
    net.add(Feedforward(shape=(hs,hs), activation='sigmoid', use_bias=True, weight_init='normal')) # w shape is (hs,hs)
    net.add(Output(shape=(os, hs), activation='sigmoid', use_bias=True, weight_init='normal')) # w shape is (os, hs)
    net.train(x,  y, loss, verbose=False, plot_loss=True)

if __name__=='__main__':
    main()