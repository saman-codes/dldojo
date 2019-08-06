# Thirdparty
import numpy as np


class Activation():
    '''
    Base class for an activation layer
    Inspired by https://github.com/eriklindernoren/ML-From-Scratch/blob/d8d86600be48117bb4d82f1f60c2d663248c0398/mlfromscratch/deep_learning/activation_functions.py
    '''

    def __call__(self, x):
        return

    def derivative(self, x):
        return


class Relu(Activation):

    def __call__(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return self.__call__(x) / x


class LeakyRelu(Activation):

    def __init__(self, mu=0.05):
        self.mu = mu
        return

    def __call__(self, x):
        return np.maximum(self.mu*x, x)

    def derivative(self, x):
        return (np.maximum(0, x) / x) * self.mu


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
        return np.exp(x) / np.exp(x).sum(axis=0, keepdims=True)

    def derivative(self, x):
        # Softmax derivative not used when CrossEntropy loss is used
        return
