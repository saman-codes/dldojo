# Standard Python
import logging

# Local
from layers import Layer
from losses import CrossEntropy

# Thirdparty
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

logging.basicConfig(format='%(message)s', level=logging.INFO)

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
        self.__name__ = 'GenericNetwork'
        self.layers = []
    
    def set_name(self, name):
        self.__name__ = name

    def train(self, x, y, loss, batch_size=1, epochs=100, learning_rate=5e-4, verbose=True, plot_loss=True):
        '''
        First implement a forward pass and store the weighted products and activations
        Then implement a backward pass, computing the gradient for each weight matrix
        and adjusting the weights using gradient descent 
        '''
        self.loss = loss
        self.training_loss = []
        data_size = x.shape[1]
        # Shuffle the original data
        s = np.random.permutation(data_size)
        x = x[:, s]
        y = y[:, s]
        for epoch in tqdm(range(epochs)):
            idx = 0
            while idx < data_size:
                if idx+batch_size <= data_size:
                    minibatch_x = x[:, idx:idx+batch_size]
                    minibatch_y = y[:, idx:idx+batch_size]
                else:
                    # If remaining data is less than size of minibatch, take all remaining data
                    minibatch_x = x[:, idx:]
                    minibatch_y = y[:, idx:]
                output = self.train_predict(minibatch_x)
                loss = self.loss(output, minibatch_y).sum()
                self.training_loss.append((epoch, loss))
                
                if verbose:
                    logging.info(f'Training loss: {loss}')

                for layer in reversed(self.layers):
                    if layer.is_output_layer:
                        dwx = layer.activation.derivative(layer.wx)
                        if isinstance(self.loss, CrossEntropy):
                            '''
                            TODO: implement CE loss for all activations, not just sigmoid
                            I think the dwx term become 1 only with sigmoid activation
                            in the output layer
                            '''
                            # CrossEntropy loss cancels out the sigma' term
                            dwx = 1
                        layer.error = self.loss.output_gradient(
                            output, minibatch_y) * dwx
                        layer.gradient = layer.error.dot(layer.x.T)
                    else:
                        layer.backward(next_layer)
                    layer.update_weights(learning_rate, batch_size)
                    next_layer = layer
                idx += batch_size

        if plot_loss:
            plt.plot([i[0] for i in self.training_loss], [i[1] for i in self.training_loss])
            plt.show()

    def test_predict(self, x):
        return self.train_predict(x, runtime='test')


    def train_predict(self, x, **kwargs):
        runtime = kwargs.get('runtime', 'train')
        # Get output shape from last layer
        os = self.layers[-1].shape
        output = np.empty(os)
        input_layer = self.layers[0]
        output = input_layer.forward(x, runtime)
        for layer in self.layers[1:]:
            output = layer.forward(output, runtime)
        return output

    def add(self, layer):
        if isinstance(layer, Layer):
            self.layers.append(layer)
        else:
            raise Exception
