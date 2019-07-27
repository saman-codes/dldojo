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
               shape = (0, 0), 
               activation = 'linear', 
               weight_init = 'xavier',
               use_bias = True):

    self.shape = shape
    self.use_bias = use_bias
    '''
    TODO: choose between this notation and the self.shape indexing notation
    atm we're using both
    '''
    # Hidden size, Input size, Channel size
    self.hs = self.shape[0]
    self.ins= self.shape[1]
    '''
    TODO: Init-ing to None for now, change to an np.zeros with correct shape
    '''
    self.error = None
    
    def _init_weights():
      if self.use_bias:
        # If using bias, add one column to the weight matrix
        self.shape = (self.hs, self.ins + 1)

      if weight_init in ['xavier', 'uniform', 'normal', 'zeros', 'ones']:
        if weight_init == 'uniform':
          self.weights = np.random.uniform(-1,1, size=self.shape)
        elif weight_init == 'normal':
          self.weights = np.random.randn(self.shape[0], self.shape[1])
        elif weight_init == 'zeros':
          self.weights = np.zeros(self.shape)
        elif weight_init == 'ones':
          self.weights = np.ones(self.shape)
      
        self.gradient = np.zeros_like(self.weights)
      
      else:
        raise Exception
        
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
    _init_activation()
        
    return
  

  def forward(self, x):
    if self.use_bias:
      try:
        # Add bias by making first row in input equal to 1
        bias_row = np.ones((1, x.shape[1]))
      except: 
        x = np.expand_dims(x, axis=1)
        bias_row = np.ones((1, x.shape[1]))
      x = np.concatenate((bias_row, x), axis=0)
    
    
    self.wx = self.weights.dot(x)
    self.out = self.activation(self.wx)
    return self.out
  
  def backward(self, next_layer):
    dwx = self.activation.derivative(self.wx)    
    self.error = next_layer.weights[:,:-1].T.dot(next_layer.error) * dwx
    
    next_layer_gradient = next_layer.error.dot(self.out.T)
    return next_layer_gradient
  

class Feedforward(Layer):
  pass


class Output(Layer):
  pass

class Softmax(Layer):
  
  def __init__(self):
    self.activation = Softmax()