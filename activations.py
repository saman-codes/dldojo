# Standard Python
import copy

# Thirdparty
import numpy as np


class Activation():
    '''
    Base class for an activation layer
    Inspired by 
    https://github.com/eriklindernoren/ML-From-Scratch/blob/master/mlfromscratch/deep_learning/activation_functions.py
    '''

    def __call__(self, x):
        return

    def derivative(self, x):
        return


class Relu(Activation):

    def __call__(self, x):
        return np.absolute(x * (x > 0))

    def derivative(self, x):
        return np.absolute(1. * (x > 0))


class LeakyRelu(Activation):

    def __init__(self, mu=0.05):
        self.mu = mu
        return

    def __call__(self, x):
        return np.maximum(self.mu*x, x)

    def derivative(self, x):
        x[x>=0] = 1
        x[x<1] = self.mu
        return x


class Linear(Activation):

    def __call__(self, x):
        return x

    def derivative(self, x):
        return 1


class Sigmoid(Activation):

    def __call__(self, x):
        return 1./(1.+np.nan_to_num((np.exp(-x))))

    def derivative(self, x):
        return self.__call__(x)*(1.-self.__call__(x))


class Softmax(Activation):

    def __call__(self, x):
        # Using normalised x for numerical stability
        norm_x = x - np.max(x, axis=0)
        return np.exp(norm_x) / np.exp(norm_x).sum(axis=0, keepdims=True)

    def derivative(self, x):
        s = self.__call__(x)
        a = s.T.dot(np.eye(s.shape[0])) - s.dot(s.T)
        a = a[:,:,np.newaxis]
        return a
        
