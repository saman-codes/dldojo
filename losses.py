import numpy as np


class Loss():
    def output_gradient(self):
        return

class MSE(Loss):
    def __call__(self, predicted, labels):
        return 0.5 * np.square(predicted - labels)

    def output_gradient(self, predicted, labels):
        return predicted - labels

class BinaryCrossEntropy(Loss):
    def __call__(self, predicted, labels):
        return - np.nan_to_num((labels*np.log(predicted) + (1-labels)*np.log(1-predicted)))

    def output_gradient(self, predicted, labels):
        return np.nan_to_num(-(labels/predicted) + (1-labels)/(1-predicted))

class CategoricalCrossEntropy(Loss):
    def __call__(self, predicted, labels):
        return -np.nan_to_num(np.sum(labels*np.log(predicted), axis=0, keepdims=True))

    def output_gradient(self, predicted, labels):
        return -np.nan_to_num(labels/predicted)
