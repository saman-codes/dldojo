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
from numpy.lib.stride_tricks import as_strided


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
        # Backward gradient (dLdout)-> the gradient backpropagated from the next layer
        self._set_backward_gradient(backward_gradient)
        # Error -> backward gradient * f'(wx)
        self._set_error()
        # Gradient (dLdw) -> error.dot(W)
        self._set_gradient()
        self._set_bias_gradient()
        # Backprop the gradient of the layer input to previous layer
        dLdx = self.weights.T.dot(self.error)
        return dLdx

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
        # Gradient is summed over batch dimension, then divided by batch_size in the optimizer
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
        self.flatten = kwargs.get('flatten', False)
        self.shape = (self.num_filters, self.input_channels*self.kernel_size**2)
        
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
        x = self._preprocessing(x)
        conv_params = dict(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        self.x = self.im2col(x, **conv_params)
        self.weights = self.kernel2row(self.weights)
        # Dot product of kernel matrix with transposed Im2Col'ed input image
        # Each row in the resulting matrix is a feature map: there is one feature map
        # per filter, and one column per window. Each element in wx is the scalar resulting 
        # from the dot product of one kernel with one window. 
        # Reshaping one row gives the feature map for one filter
        _, num_windows, bs = self.x.shape
        num_filters, _ = self.weights.shape
        self.wx = np.zeros(shape=(num_filters, num_windows, bs))
        for b, batch_el in enumerate(self.x.T):
            self.wx[:,:,b] = self.weights.dot(batch_el.T)
        if self.flatten:
            self.wx = self.wx.reshape(-1, bs)
        else:
            # if not flattening the output, reshape it as a multichannel feature map
            self.wx = self.col2im(self.wx, **conv_params)
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
        # Gradient of loss wrt the conv layer weights (filters) is a convolution of 
        # the error (dLdwx * f'(wx)) on the input x, recast as dot product with im2col
        gradient_shape = (self.num_filters, self.kernel_size**2, bs)
        self.gradient = np.zeros(shape=gradient_shape)
        for b, batch_el in enumerate(self.x.T):
            self.gradient[:,:,b] = self.error[:,:,b].dot(batch_el)
        self.gradient = self.gradient.mean(axis=2)
        return

    def _set_error(self):
        dwx = self.activation.derivative(self.wx)
        # if self.add_dropout:
        #     dwx *= self.dropout_mask
        # if len(self.backward_gradient.shape) > 2:
        #     kwargs = dict(backwards=True, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        #     self.backward_gradient = self.col2im(self.backward_gradient, **kwargs)
        self.error = self.backward_gradient * dwx
        return
    
    @staticmethod
    def im2col(x, kernel_size=3, stride=1, padding=0):
        '''
        Reshape a multichannel input image or feature map into a single-channel matrix, 
        where each column is a window obtained by sliding a filter over the image, and there is
        one row entry per element in the kernel
        '''
        if len(x.shape) < 4: 
            num_channels, side_squared, bs = x.shape
            side = int(np.sqrt(side_squared))
            x = x.reshape(num_channels, side, side, bs)
        
        c, side, _, bs = x.shape
        num_windows = int((side-kernel_size+2*padding)/stride)+1
        tot_num_windows = int(num_windows**2)
        # Each window has size (ks,ks) and there num_windows patches in each direction
        # (e.g. a 3x3 filter on a 4x4 image defines 3 windows horizontally and 3 windows vertically, 
        # for a total of 9 windows. So the output array has shape (ks, ks, num_windows, num_windows)
        shape = (kernel_size, kernel_size, num_windows, num_windows) 
        im2c = np.zeros(shape=(c*kernel_size**2, tot_num_windows, bs)) 
        for b, batch_el in enumerate(x.T):
            for c, channel in enumerate(batch_el.T):
                im2c[c*kernel_size**2:(c+1)*kernel_size**2,:,b] = as_strided(channel, shape=shape, strides=channel.strides*2).reshape(kernel_size**2, -1)
        return im2c
    
    
    # def col2im(self, x, backwards=False, kernel_size=3, stride=1, padding=0):
    #     '''
    #     Reverse operation of im2col: turn a multichannel feature map into a corresponding image:
    #     each column (the result of applying the convolution kernel over a single wind\ow in the input) 
    #     is reshaped into the shape of the input image the filters have been convolved on
    #     Output is then of size (num_filters, input_side, input_side, batch_size)
    #     '''
    #     h, w = (28,28)
    #     num_filters = 6

    #     if len(x.shape) == 2:
    #         x = x[np.newaxis, :,:]
    #     feature_maps_channels, num_windows, bs = x.shape
    #     nw = np.sqrt(num_windows)
    #     # kernel_size = int(h + 2*padding + (1-nw)*stride)
    #     h = int(self.kernel_size - 2*padding - (1-nw)*stride)
    #     w = h
    #     c2im = np.zeros(shape=(num_filters, h, w, bs))
    #     # for b, batch_el in enumerate(x.T):
    #     #     i, col_idx =(0,0)
    #     #     while i+kernel_size <= w:
    #     #         j=0
    #     #         while j+kernel_size <= h:
    #     #             for k in range(feature_maps_channels):
    #     #                 if backwards:
    #     #                     # When c2im is applied in the backward pass, we need to sum the gradients for repeated elements
    #     #                     c2im[:, i:i+kernel_size, j:j+kernel_size, b] += batch_el[k*kernel_size**2:(k+1)*kernel_size**2, col_idx].reshape(kernel_size, kernel_size)
    #     #                 else:
    #     #                     # otherwise just assign the elements
    #     #                     try:
    #     #                         c2im[:, i:i+kernel_size, j:j+kernel_size, b] = batch_el[k*kernel_size**2:(k+1)*kernel_size**2, col_idx].reshape(kernel_size, kernel_size)
    #     #                     except:
    #     #                         pass
    #     #             j += stride
    #     #         col_idx += 1
    #     #         i += stride
    #     for b, batch_el in enumerate(x.T):
    #         for c, channel in enumerate(batch_el.T):
    #             if backwards:
    #                  # When c2im is applied in the backward pass, we need to sum the gradients for repeated elements
    #                 im2c[c*kernel_size**2:(c+1)*kernel_size**2,:,b] += as_strided(channel, shape=shape, strides=channel.strides*2).reshape(kernel_size**2, -1)
    #             else:
    #                 # otherwise just assign the elements
    #                 im2c[c*kernel_size**2:(c+1)*kernel_size**2,:,b] = as_strided(channel, shape=shape, strides=channel.strides*2).reshape(kernel_size**2, -1)
    #     return c2im

    @staticmethod
    def kernel2row(kernel):
        '''
        Reshape a 3-d volume kernel into a 2-d volume matrix
        Each volume filter becomes a row in this matrix
        '''
        return kernel.reshape(-1, kernel.shape[-1])