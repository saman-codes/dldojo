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
                 **kwargs,
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
                self.mean_wx_avg, self.var_wx_avg = (0, 0)
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

    def backward(self, backward_gradient):
        self._set_backward_gradient(backward_gradient)
        self._set_error()
        self._set_gradient()
        self._set_bias_gradient()
        return self.weights.T.dot(self.error)

    def _apply_dropout(self, runtime):
        if runtime == 'train':
            self._set_dropout_mask()
            self.out *= self.dropout_mask
        return self.out

    def _apply_batch_normalization(self, runtime, momentum=0.99):
        # We batch normalization before nonlinearity, so we add it
        # inside the layer, instead of creating a new layer
        # Bias is included in beta parameter later, so not here
        if runtime=='train':
            # Mean and variance are estimated over batches
            self.mean_wx = self.wx.mean(axis=1, keepdims=True)
            self.var_wx = ((self.wx - self.mean_wx)**2).mean(axis=1, keepdims=True)
            # Keep running average of mean and variance for use in testing
            self.mean_wx_avg = momentum*self.mean_wx_avg + (1-momentum)*self.mean_wx
            self.var_wx_avg = momentum*self.var_wx_avg + (1-momentum)*self.var_wx
            # Get normalised wx - Using var names from batchnorm paper
            self.x_hat = (self.wx - self.mean_wx) / (np.sqrt(self.var_wx) + 1e-12)
            # Multiply by alpha and add beta parameter
            self.y = self.batchnorm_gamma * self.x_hat + self.batchnorm_beta
        else:
            # For testing, use collected moving averages
            self.x_hat = (self.wx - self.mean_wx_avg) / (np.sqrt(self.var_wx_avg) + 1e-12)
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
        # Save gamma parameter before updating it - use for backpropagation
        gamma = self.batchnorm_gamma
        self.batchnorm_gamma = self.batchnorm_optimizer.update_weights(
                self.batchnorm_gamma, learning_rate, batch_size, self.batchnorm_gamma_gradient
                )
        self.batchnorm_beta = self.batchnorm_optimizer.update_weights(
                self.batchnorm_beta, learning_rate, batch_size, self.batchnorm_beta_gradient
                )
        # Update layer error
        inverse_std = 1. / (np.sqrt(self.var_wx) + 1e-12)
        self.error = 1. / batch_size  * gamma * inverse_std * (
                    - self.batchnorm_gamma_gradient * self.x_hat
                    + batch_size * self.error - self.batchnorm_beta_gradient
                    )
        # Not updating bias gradient since bias is not used
        self._set_gradient()
        return

    def _set_batchnorm_gradients(self):
        self.batchnorm_gamma_gradient = (self.error * self.x_hat).sum(axis=1, keepdims=True)
        self.batchnorm_beta_gradient = self.error.sum(axis=1, keepdims=True)
        return
    
    def _set_backward_gradient(self, backward_gradient):
        self.backward_gradient = backward_gradient
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
        if len(dwx.shape) == 2:
            # If d(wx) is elementwise, use hadamard product
            self.error = self.backward_gradient * dwx
        else:
            # else use dot product of each col (one batch element) with the matrices 
            # on the diagonal of the full jacobian (off-diagonal entries are zero matrices)
            # Used for categorical cross entropy and non-flattened conv layer backprop
            bs = self.backward_gradient.shape[-1]
            self.error = np.array(
                [self.backward_gradient[:,i].dot(dwx[:,:,i]) for i in range(bs)],
            ).T
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
        self.num_filters = kwargs.get('num_filters', 12)
        self.kernel_size = kwargs.get('kernel_size', 3)
        self.input_channels = kwargs.get('input_channels', 1)
        self.stride = kwargs.get('stride', 1)
        self.padding = kwargs.get('padding', 0)
        self.shape = (self.num_filters, self.input_channels*self.kernel_size**2)
        self.flatten = kwargs.get('flatten', False)
        
        super().__init__(
            shape=self.shape,
            use_bias=False, 
            **kwargs
            )
        return

    def forward(self, x, runtime):
        if len(x.shape) == 2:
            # Add a new dimension for single-channel inputs
            x = np.expand_dims(x, 0)
        bs = x.shape[-1]
        self.x = self._preprocessing(x)
        # Apply Im2Col operation
        self.x = self.im2col(
            self.x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        self.weights = self.kernel2row(self.weights)
        # Dot product of kernel matrix with transposed Im2Col'ed input image
        # Each row in the resulting matrix is a feature map: there is one feature map
        # per filter, and one column per window. Each element in wx is the scalar resulting 
        # from the dot product of one kernel with one window. 
        # Reshaping one row gives the feature map for one filter
        self.wx = np.moveaxis(np.array([self.weights.dot(self.x[:,:,b]) for b in range(bs)]), 0, -1)
        if self.flatten:
            self.wx = self.wx.reshape(-1, bs)
        self.out = self.activation(self.wx)
        return self.out
    
    def backward(self, backward_gradient):
        self._set_backward_gradient(backward_gradient)
        self._set_error()
        self._set_gradient()
        self._set_bias_gradient()
        bs = self.error.shape[-1]
        return np.moveaxis(np.array([self.weights.T.dot(self.error[:,:,b]) for b in range(bs)]), 0, -1)
    
    def _set_gradient(self):
        _, num_windows, bs = self.x.shape
        # self.error is dL/d_feature_map
        if self.flatten:
                self.error = self.error.reshape((self.num_filters, num_windows, bs))
        self.gradient =  np.array([self.error[:,:,b].dot(self.x[:,:,b].T) for b in range(bs)]).mean(axis=0)
        return

    @staticmethod
    def im2col(x, kernel_size=3, stride=1, padding=0):
        '''
        Reshape a multichannel input image or feature map into a single-channel matrix, 
        where each column is a window obtained by sliding a filter over the image, and there is
        one row entry per element in the kernel
        '''
        num_channels, side_squared, bs = x.shape
        side = int(np.sqrt(side_squared))
        def _get_col(img, **kwargs):
            ks = kwargs.get('kernel_size')
            p = kwargs.get('padding')
            s = kwargs.get('stride')
            c, h, w = img.shape
            num_windows = int((side+2*p-ks/(s)+1)**2)
            im2c = np.zeros(shape=(c*ks**2, num_windows))
            i,col_idx =(0,0)
            while i+ks <= w:
                j=0
                while j+ks <= h:
                    for k in range(c):
                        im2c[k*ks**2:(k+1)*ks**2, col_idx] = img[k, i:i+ks, j:j+ks].reshape(-1,)
                    col_idx += 1
                    j += s
                i += s
            return im2c

        x_r = x.reshape(num_channels, side, side, bs)
        kwargs = dict(kernel_size=kernel_size, padding=padding, stride=stride)
        im2c = np.moveaxis(np.array([_get_col(x_r[:,:,:,b], **kwargs) for b in range(bs)]), 0,-1)
        return im2c

    @staticmethod
    def kernel2row(kernel):
        '''
        Reshape a 3-d volume kernel into a 2-d volume matrix
        Each volume filter becomes a row in this matrix
        '''
        return kernel.reshape(-1, kernel.shape[-1])