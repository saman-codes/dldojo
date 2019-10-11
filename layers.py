# Python
import string

# Local
import settings
import optimizers
from activations import *
from operations import Operations
from initializers import Initializer

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
                 batch_normalization=False,
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
        self.batch_normalization = batch_normalization
        self.optimizer = None

        def _init_weights():
            self.weights = Initializer.initialize_weights(weight_init, self.shape)
            if self.is_trainable:
                self.gradient = np.zeros_like(self.weights)
            if self.batch_normalization:
                self.bn_gamma = np.ones((self.shape[0], 1))
                self.bn_beta = np.zeros((self.shape[0], 1))
                self.mean_wx_list = list()
                self.var_wx_list = list()

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
        if self.batch_normalization:
            # Apply batch normalization before nonlinearity
            # Bias is included in beta parameter later,
            # so no bias is applied here
            if runtime=='train':
                # Mean and variance are estimated over batches
                mean_wx = self.wx.mean(axis=1, keepdims=True)
                var_wx = ((self.wx - mean_wx)**2).mean(axis=1, keepdims=True)
                # Collect mean and variance for later use in testing
                self.mean_wx_list.append(mean_wx)
                self.var_wx_list.append(var_wx)
                # Get normalised wx
                norm_x = (self.wx - mean_wx) / (np.sqrt(var_wx) + 1e-8)
                # Multiply by alpha and add beta parameter
                self.wx = self.bn_gamma * norm_x + self.bn_beta
            else:
                # During inference use population estimators
                mean_wx = np.mean(self.mean_wx_list)
                var_wx = np.mean(self.var_wx_list)
                norm_x = (self.wx - mean_wx) / (np.sqrt(var_wx) + 1e-8)
                self.wx = self.bn_gamma * norm_x + self.bn_beta
        else:
            # No batch normalization applied
            if self.use_bias:
                self.wx += self.bias.dot(np.ones((1,self.wx.shape[1])))
        # Apply activation function
        self.out = self.activation(self.wx)
        # Apply dropout
        if self.add_dropout and runtime == 'train':
            self._set_dropout_mask()
            self.out *= self.dropout_mask
        return self.out

    def backward(self):
        self._set_error()
        self._set_gradient()
        self._set_bias_gradient()
        return

    def update_weights(self, learning_rate, batch_size):
        if self.is_trainable:
            self.weights = self.optimizer.update_weights(self.weights, learning_rate, batch_size, self.gradient)
            if self.use_bias:
                self.bias = self.optimizer.update_bias(self.bias, learning_rate, batch_size, self.bias_gradient)
        return

    def set_next_layer(self, next_layer):
        self.next_layer = next_layer
        return

    def _set_error(self):
        dwx = self.activation.derivative(self.wx)
        if self.add_dropout:
            dwx *= self.dropout_mask
        self.error = self.next_layer.weights.T.dot(self.next_layer.error) * dwx
        return

    def _set_gradient(self):
        self.gradient = self.error.dot(self.x.T)
        return

    def _set_bias_gradient(self):
        self.bias_gradient = self.error.sum(axis=1, keepdims=True)
        return

    def _set_dropout_mask(self):
        # This mask implements Inverted Dropout
        self.dropout_mask = (np.random.rand(*self.out.shape) < self.dropout_p) / self.dropout_p
        return

    def set_optimizer(self, optimizer):
        # Optimizer has to be set at the layer level, since some algorithms
        # (e.g. Adam) use layer-level gradient caches
        if not self.optimizer:
            opt_class_name= ''.join([w.capitalize() for w in optimizer.split('_')])
            try:
                opt_class = getattr(optimizers, opt_class_name)
                self.optimizer = opt_class()
            except:
                raise Exception(f'Optimizer {opt_class_name} is not a valid optimizer')
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