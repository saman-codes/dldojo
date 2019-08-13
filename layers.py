# Local
import settings
from activations import Linear, Sigmoid, Relu, LeakyRelu, Softmax

# Thirdparty
import numpy as np


class Layer():
    '''
    Base class for a neural network layer
    Shape is (hidden size, input size, num channels)
    '''

    def __init__(self,
                 shape=(0, 0),
                 activation='linear',
                 weight_init='xavier',
                 use_bias=True,
                 bias_init='zeros',
                 is_trainable=True,
                 dropout=1.,
                 flatten=False,
                 minmax_scaling=False):

        self.is_trainable = is_trainable
        self.shape = shape
        self.use_bias = use_bias
        self.add_dropout = bool(dropout < 1.)
        self.dropout_p = dropout
        self.flatten = flatten
        self.minmax_scaling = minmax_scaling
        self.is_output_layer = False
        '''
        TODO: Init-ing to None for now, change to an np.zeros with correct shape
        '''
        self.error = None
        def _init_weights():
            if weight_init in settings.WEIGHTS_INITIALIZERS:
                if weight_init == 'uniform':
                    self.weights = np.random.uniform(-1, 1, size=self.shape)
                elif weight_init == 'normal':
                    self.weights = np.random.randn(*self.shape)
                elif weight_init == 'zeros':
                    self.weights = np.zeros(self.shape)
                elif weight_init == 'ones':
                    self.weights = np.ones(self.shape)
                elif weight_init == 'glorot_normal':
                    sigma = np.sqrt(2/(self.shape[0] + self.shape[1]))
                    self.weights = np.random.randn(*self.shape) * sigma
                elif weight_init == 'glorot_uniform':
                    limit = np.sqrt(6/(self.shape[0] + self.shape[1]))
                    self.weights = np.random.uniform(-limit, +limit, self.shape)
                elif weight_init == 'he_normal':
                    sigma = np.sqrt(2/(self.shape[1]))
                    self.weights = np.random.randn(*self.shape) * sigma
                elif weight_init == 'he_uniform':
                    limit = np.sqrt(6/(self.shape[1]))
                    self.weights = np.random.uniform(-limit, +limit, self.shape)


            if self.is_trainable:
                self.gradient = np.zeros_like(self.weights)

            else:
                raise Exception

        def _init_bias():
            bias_shape = (self.shape[0], 1)
            self.bias = np.zeros(bias_shape)
            if use_bias and bias_init in settings.WEIGHTS_INITIALIZERS:
                if bias_init == 'uniform':
                    self.bias = np.random.uniform(-1, 1, size=bias_shape)
                elif bias_init == 'normal':
                    self.bias = np.random.randn(*bias_shape)
                elif bias_init == 'zeros':
                    self.bias = np.zeros(bias_shape)
                elif bias_init == 'ones':
                    self.bias = np.ones(bias_shape)
                elif weight_init == 'glorot_normal':
                    sigma = np.sqrt(2/(self.shape[0] + self.shape[1]))
                    self.weights = np.random.randn(*self.shape) * sigma
                elif weight_init == 'glorot_uniform':
                    limit = np.sqrt(6/(self.shape[0] + self.shape[1]))
                    self.weights = np.random.uniform(-limit, +limit, self.shape)
                elif weight_init == 'he_normal':
                    sigma = np.sqrt(2/(self.shape[1]))
                    self.weights = np.random.randn(*self.shape) * sigma
                elif weight_init == 'he_uniform':
                    limit = np.sqrt(6/(self.shape[1]))
                    self.weights = np.random.uniform(-limit, +limit, self.shape)

        def _init_activation():
            if activation in ['relu', 'leaky_relu', 'linear', 'sigmoid', 'softmax']:
                if activation == 'relu':
                    self.activation = Relu()
                elif activation == 'linear':
                    self.activation = Linear()
                elif activation == 'leaky_relu':
                    self.activation = LeakyRelu()
                elif activation == 'sigmoid':
                    self.activation = Sigmoid()
                elif activation == 'softmax':
                    self.activation = Softmax()
            else:
                raise Exception

        _init_weights()
        _init_bias()
        _init_activation()

        return

    def forward(self, x, runtime='train'):
        if self.flatten:
            x = x.reshape(-1, 1)
        self.x = x
        if self.minmax_scaling:
            self.x = np.divide((x - np.amin(x)),(np.amax(x) - np.amin(x)))
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

    def update_weights(self, learning_rate, batch_size):
        if self.is_trainable:
            # Update weights
            self.weights -= learning_rate/batch_size * self.gradient
            # Update bias (average error over batches)
            if self.use_bias:
                self.bias -= learning_rate/batch_size * self.bias_gradient
        return

    def _set_dropout_mask(self):
        # This mask implements Inverted Dropout
        self.dropout_mask = (np.random.rand(*self.out.shape) < self.dropout_p) / self.dropout_p
        return

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

class BatchNormalization(Layer):
    pass