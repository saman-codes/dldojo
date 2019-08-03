# Local
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
                 is_trainable=True):
        
        self.is_trainable = is_trainable
        self.shape = shape
        self.use_bias = use_bias
        self.is_output_layer = False
        '''
        TODO: Init-ing to None for now, change to an np.zeros with correct shape
        '''
        self.error = None

        def _init_weights():
            if weight_init in ['xavier', 'uniform', 'normal', 'zeros', 'ones']:
                if weight_init == 'uniform':
                    self.weights = np.random.uniform(-1, 1, size=self.shape)
                elif weight_init == 'normal':
                    self.weights = np.random.randn(
                        self.shape[0], self.shape[1])
                elif weight_init == 'zeros':
                    self.weights = np.zeros(self.shape)
                elif weight_init == 'ones':
                    self.weights = np.ones(self.shape)
                elif weight_init == 'xavier':
                    '''
                    TODO: implement xavier initialisation
                    '''
                    self.weights = np.random.uniform(-1, 1, size=self.shape)
            
            if self.is_trainable:
                self.gradient = np.zeros_like(self.weights)

            else:
                raise Exception

        def _init_bias():
            bias_shape = (self.shape[0], )
            self.bias = np.zeros(bias_shape)
            if use_bias and bias_init in ['uniform', 'normal', 'zeros', 'ones']:
                if bias_init == 'uniform':
                    self.bias = np.random.uniform(-1, 1, size=bias_shape)
                elif bias_init == 'normal':
                    self.bias = np.random.randn(bias_shape[0], )
                elif bias_init == 'zeros':
                    self.bias = np.zeros(bias_shape)
                elif bias_init == 'ones':
                    self.bias = np.ones(bias_shape)

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
                    self.activation == Softmax()
            else:
                raise Exception

        _init_weights()
        _init_bias()
        _init_activation()

        return

    def forward(self, x):
        self.x = x
        self.wx = self.weights.dot(x)
        self.out = self.activation(self.wx)
        return self.out

    def backward(self, next_layer):
        '''
        Calculate current layer error and next layer gradient
        '''
        dwx = self.activation.derivative(self.wx)
        self.error = next_layer.weights.T.dot(next_layer.error) * dwx
        next_layer.gradient = next_layer.error.dot(self.out.T)
        return
    
    def update_weights(self, learning_rate, batch_size):
        # Update weights
        self.weights -= learning_rate/batch_size * self.gradient
        # Update bias (average error over batches)
        if self.use_bias:
            self.bias -= learning_rate / \
                batch_size * self.error.sum(axis=1)
        return

class Input(Layer):
    def __init__(self, input_shape=None):
        super().__init__( 
                    shape=input_shape,
                    activation='linear',
                    weight_init='ones',
                    use_bias=False,
                    is_trainable=False)
        return

class Feedforward(Layer):
    pass

class Output(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_output_layer = True
        return

class Convolutional(Layer):
    pass

class BatchNormalization(Layer):
    pass

class Dropout(Layer):
    '''
    This layer implements Inverted Dropout, which 
    scales up the weights during training instead 
    of scaling them down at test time
    See: 
    '''
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        p = kwargs.get('p', 0.5)
        self.is_trainable = False
        # p = probability the neuron is active
        self.p = p 
        self.weights = None
        return

    def forward(self, x):
        self.dropout_mask = np.zeros_like(x)
        idx = np.random.sample(x.shape[0])
        idx[idx>=(1-self.p)] = 1
        idx = np.where(idx == 1)
        self.dropout_mask[idx] = 1/self.p
        self.out =  x * self.dropout_mask
        return self.out 

    def backward(self, next_layer):
        pass

        

        
