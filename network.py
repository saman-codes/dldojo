# Local
from layers import Layer
from losses import CrossEntropy

# Thirdparty
import numpy as np
import matplotlib.pyplot as plt

class Network():
  '''
  Base class for a neural network model
  '''
  
  def __init__(self):
    '''
    TODO: implement: layers stored in a dictionary, defining a computation graph
    self.layers = {}  
    For now, store in a list
    '''
    self.layers = []
        
  def train(self, x, labels, loss, epochs=100, learning_rate=5e-4, verbose=True, plot_loss=True):
    '''
    First implement a forward pass and store the weighted products and activations
    Then implement a backward pass, computing the gradient for each weight matrix
    and adjusting the weights using gradient descent 
    '''
    self.loss = loss
    self.training_loss = []
    batch_size = x.shape[1]
    '''
    TODO: add code to sample minibatch and train for n epochs
    TODO: add code to shuffle data and then feed in order, instead of sampling 
    '''
    for epoch in range(epochs):
      output = self.predict(x)
      loss = self.loss(output, labels).sum()
      self.training_loss.append((epoch, loss))
      if verbose:
          print(f'Training loss: {loss}')
          
      for i, layer in enumerate(reversed(self.layers)):
        if i == 0:
          dwx = layer.activation.derivative(layer.wx)  
          if isinstance(self.loss, CrossEntropy):
            # CrossEntropy loss cancels out the sigma' term
            dwx = 1  
          layer.error = self.loss.output_gradient(output, labels) * dwx
        else:
          next_layer.gradient = layer.backward(next_layer)
          # Update weights
          next_layer.weights[:,:-1] -= learning_rate/batch_size * next_layer.gradient
          # Update bias (sum error over batches)
          next_layer.weights[:,-1] -= learning_rate/batch_size * next_layer.error.sum(axis=1)          
        next_layer = layer
    
    if plot_loss:
      plt.plot([i[0] for i in self.training_loss], [i[1] for i in self.training_loss])
      plt.show()
    return
  
  def predict(self, x):
    '''
    Get input shape from first layer and output shape from last layer
    '''
    os = self.layers[-1].shape
    output = np.empty(os) 
    for i, layer in enumerate(self.layers):
      if i == 0:
        output = self.layers[i].forward(x)
      else:
        output = self.layers[i].forward(output)
    return output
  
  def add(self, layer):
    if isinstance(layer, Layer):
      self.layers.append(layer)
    else:
      raise Exception
    return
  
  