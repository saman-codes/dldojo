# Python
import string

# Local
import settings
import optimizers
from operations import Operations
from initializers import Initializer
from activations import *

# Thirdparty
import numpy as np


class Layer():
    '''
    Base class for a neural network layer
    Shape is (hidden size, input size)
    '''

    def __init__(self,
                 shape=(0, 0),
                 activation='sigmoid',
                 weight_init='glorot_uniform',
                 use_bias=True,
                 bias_init='zeros',
                 is_trainable=True,
                 dropout=1.,
                 preprocessing=list(),
    ):

        self.is_trainable = is_trainable
        self.shape = shape
        self.use_bias = use_bias
        self.add_dropout = bool(dropout < 1.)
        self.dropout_p = dropout
        self.is_output_layer = False
        self.preprocessing = preprocessing
        self.error = None

        def _init_weights():
            self.weights = Initializer.initialize_weights(weight_init, self.shape)
            if self.is_trainable:
                self.gradient = np.zeros_like(self.weights)

        def _init_bias():
            bias_shape = (self.shape[0], 1)
            self.bias = np.zeros(bias_shape)
            if use_bias:
                self.bias = Initializer.initialize_weights(bias_init, bias_shape)

        def _init_activation():
            self.activation = Initializer.initialize_activation(activation)

        _init_weights()
        _init_bias()
        _init_activation()

        return

    def forward(self, x, runtime='train'):
        self.x = self._preprocessing(x)
        self.wx = self.weights.dot(self.x)
        if self.use_bias:
            self.wx += self.bias.dot(np.ones((1,self.wx.shape[1])))
        self.out = self.activation(self.wx)
        if self.add_dropout and runtime == 'train':
            self._set_dropout_mask()
            self.out *= self.dropout_mask
        return self.out

    def backward(self, next_layer):
        '''
        Calculate current layer error and gradient
        '''
        dwx = self.activation.derivative(self.wx)
        if self.add_dropout:
            dwx *= self.dropout_mask
        self.error = next_layer.weights.T.dot(next_layer.error) * dwx
        self.gradient = self.error.dot(self.x.T)
        self.bias_gradient = self.error.sum(axis=1, keepdims=True)
        return

    def update_weights(self, learning_rate, batch_size, optimizer=''):
        if self.is_trainable:
            opt_class_name= ''.join([w.capitalize() for w in optimizer.split('_')])
            try:
                opt_class = getattr(optimizers, opt_class_name)
                optimizer = opt_class()
                self.weights = optimizer.update_weights(self.weights, learning_rate, batch_size, self.gradient)
                if self.use_bias:
                    self.bias = optimizer.update_bias(self.bias, learning_rate, batch_size, self.bias_gradient)
            except:
                raise Exception(f'Optimizer {opt_class_name} is not a valid optimizer')
        return

    def _set_dropout_mask(self):
        # This mask implements Inverted Dropout
        self.dropout_mask = (np.random.rand(*self.out.shape) < self.dropout_p) / self.dropout_p
        return

    def _preprocessing(self, x):
        self.x = x
        for p in self.preprocessing:
            try:
                pp_step = getattr(Operations, p)
                self.x = pp_step(self.x)
            except:
                raise Exception(f'{p} is not a valid preprocessing step')
        return self.x

class Feedforward(Layer):
    pass

class Output(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_output_layer = True
        return

class Convolutional(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Check that layer has at least 3 dimensions
        assert(len(self.shape) > 2)
        return

    def forward():
        return

    def backward():
        return