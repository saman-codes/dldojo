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
            if self.batch_normalization:
                # Initialize gamma to 0 and beta to 1
                self.batchnorm_gamma = np.ones((self.shape[0], 1))
                self.batchnorm_beta = np.zeros((self.shape[0], 1))
                self.mean_wx_avg = 0
                self.var_wx_avg = 0
                self.batchnorm_optimizer = None

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
            self._apply_batch_normalization(runtime)
            # Apply activation function to normalized output
            self.out = self.activation(self.y)
        else:
            # No batch normalization applied
            if self.use_bias:
                self.wx += self.bias.dot(np.ones((1,self.wx.shape[1])))
            # Apply activation function
            self.out = self.activation(self.wx)
        # Apply dropout
        if self.add_dropout:
            self.out = self._apply_dropout(runtime)
        return self.out

    def backward(self):
        self._set_error()
        self._set_gradient()
        self._set_bias_gradient()
        return
    
    def _apply_dropout(self, runtime):
        if runtime == 'train':
            self._set_dropout_mask()
            self.out *= self.dropout_mask
        return self.out
    
    def _apply_batch_normalization(self, runtime):
        # Apply batch normalization before nonlinearity
        # Bias is included in beta parameter later, so not here
        if runtime=='train':
            # Mean and variance are estimated over batches
            self.mean_wx = self.wx.mean(axis=1, keepdims=True)
            self.var_wx = ((self.wx - self.mean_wx)**2).mean(axis=1, keepdims=True)
            # Collect mean and variance for later use in testing
            self.mean_wx_avg += self.mean_wx
            self.mean_wx_avg /= 2
            self.var_wx_avg += self.var_wx
            self.var_wx_avg /= 2
            # Get normalised wx - Using var names from batchnorm paper
            self.x_hat = (self.wx - self.mean_wx) / (np.sqrt(self.var_wx) + 1e-8)
            # Multiply by alpha and add beta parameter
            self.y = self.batchnorm_gamma * self.x_hat + self.batchnorm_beta
        else:
            # During inference use population estimators
            self.x_hat = (self.wx - self.mean_wx_avg) / (np.sqrt(self.var_wx_avg) + 1e-8)
            self.y = self.batchnorm_gamma * self.x_hat + self.batchnorm_beta
        return

    def update_weights(self, learning_rate, batch_size):
        if self.is_trainable:
            if self.batch_normalization:
                self._update_batchnorm_parameters(learning_rate, batch_size)
            self.weights = self.optimizer.update_weights(
                self.weights, learning_rate, batch_size, self.gradient
                )
            if self.use_bias:
                self.bias = self.optimizer.update_bias(
                    self.bias, learning_rate, batch_size, self.bias_gradient
                    )
        return
    
    def _update_batchnorm_parameters(self, learning_rate, batch_size):
        if not(self.batchnorm_optimizer):
                self._set_batchnorm_optimizer()
        self._set_batchnorm_gradients()
        self.batchnorm_gamma = self.batchnorm_optimizer.update_weights(
                self.batchnorm_gamma, learning_rate, batch_size, self.batchnorm_gamma_gradient
                )
        self.batchnorm_beta = self.batchnorm_optimizer.update_weights(
                self.batchnorm_beta, learning_rate, batch_size, self.batchnorm_beta_gradient
                )
        # Update layer error
        # dxhat = self.batchnorm_gamma * self.next_layer.error
        inverse_std = 1. / (np.sqrt(self.var_wx) + 1e-8)
        # self.error = (1. / batch_size) * inverse_std * (
        #             batch_size * dxhat - dxhat.sum(axis=1, keepdims=True) -
        #             self.xhat * (dxhat * self.xhat).sum(axis=1, keepdims=True)
        #         )
        a = 1. / batch_size  * self.batchnorm_gamma * inverse_std
        dLdgamma = self.error.dot(self.x_hat.T).sum(axis=1, keepdims=True)
        dLdbeta = self.error.sum(axis=1, keepdims=True)
        self.error = a * ( - dLdgamma * self.x_hat + batch_size * 
                    self.error.sum(axis=1, keepdims=True) - dLdbeta
                    )
        
        # Not updating bias gradient since bias is not used 
        self._set_gradient()
        return
    
    def _set_batchnorm_gradients(self):
        self.batchnorm_gamma_gradient = self.error.dot(self.x_hat.T).sum(axis=1, keepdims=True)
        self.batchnorm_beta_gradient = self.error.sum(axis=1, keepdims=True)
        return
    
    def set_next_layer(self, next_layer):
        self.next_layer = next_layer
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

    def _set_error(self):
        dwx = self.activation.derivative(self.wx)
        if self.add_dropout:
            dwx *= self.dropout_mask
        # Error <- next layer error * dwx
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

    def _set_batchnorm_optimizer(self):
        self.batchnorm_optimizer = self.optimizer.__class__()
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