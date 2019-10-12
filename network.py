# Standard Python
import os
import copy
import pickle
import logging
from collections import OrderedDict

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
    def __init__(self,
        load_weights=False,
        ):
        self.__name__ = 'GenericNetwork'
        self.layers = list()
        self.load_weights = load_weights

    def set_name(self, name):
        self.__name__ = name

    def train(self, x, y, loss,
                batch_size=1,
                epochs=100,
                learning_rate=5e-4,
                optimizer='minibatch_sgd',
                regularizer=None,
                verbose=False,
                plot_loss=False,
                shuffle_data=True,
                gradient_check=False,
                save_weights=False,
            ):
        '''
        First implement a forward pass and store the weighted products and activations
        Then implement a backward pass, computing the gradient for each weight matrix
        and adjusting the weights using gradient descent
        '''
        self.loss = loss
        self.training_loss = []
        self.data_size = x.shape[1]
        self.batch_size = batch_size
        self.set_optimizer(optimizer)

        if regularizer is not None:
            assert(isinstance(regularizer, tuple))
            self.regularizer, self.reg_lambda = regularizer
        else:
            self.regularizer = regularizer
        if shuffle_data:
            # Shuffle the original data
            s = np.random.permutation(self.data_size)
            x = x[:, s]
            y = y[:, s]
        if self.load_weights and os.path.exists(load_weights):
            try:
                weights_list = pickle.load(load_weights)
                for idx, l in enumerate(self.layers):
                    l.weights = weights_list[idx][0]
                    l.bias = weights_list[idx][1]
            except Exception as ex:
                logging.exception('Failed to load weights')

        for epoch in tqdm(range(epochs)):
            self.idx = 0
            while self.idx < self.data_size:
                minibatch_x, minibatch_y = self.get_minibatch(x, y)
                # Forward pass
                output = self.train_predict(minibatch_x)
                loss = self.loss(output, minibatch_y).mean()

                if verbose:
                    logging.info(f'Training loss: {loss}')

                # Training step
                for layer in reversed(self.layers):
                    if layer.is_output_layer:
                        dwx = layer.activation.derivative(layer.wx)
                        layer.error = self.loss.output_gradient(
                            output, minibatch_y) * dwx
                        layer.gradient = layer.error.dot(layer.x.T)
                        layer.bias_gradient = layer.error.sum(axis=1, keepdims=True)
                    else:
                        layer.set_next_layer(next_layer)
                        layer.backward()
                    if self.regularizer:
                        self._add_regularization_term(layer)
                    if gradient_check:
                        self._check_gradient(minibatch_x, minibatch_y, layer)
                    else:
                        # Do not update the weights if checking the gradients
                        layer.update_weights(learning_rate, self.batch_size)
                    next_layer = layer

            # Save training loss at end of epoch for plotting
            self.training_loss.append((epoch, loss))

        if plot_loss:
            plt.plot([i[0] for i in self.training_loss], [i[1] for i in self.training_loss])
            plt.show()

        if save_weights:
            weights_list = [(l.weights, l.bias) for l in self.layers]
            fname = os.path.join(os.getcwd(), f'{self.__name__}_{epoch}')
            with open(fname, 'w+') as f:
                os.chmod(fname, 777)
                pickle.dump(weights_list, fname)
                logging.info(f'Weights saved on file {fname}')

    def _add_regularization_term(self, layer):
        if self.regularizer == 'L2':
            # Derivative of the squared weights, so we lose the power of 2
            d_reg_term = layer.weights
        layer.gradient += self.reg_lambda * d_reg_term/d_reg_term.size
        return

    def test_predict(self, x):
        return self.train_predict(x, runtime='test')

    def train_predict(self, x, **kwargs):
        runtime = kwargs.get('runtime', 'train')
        # Get output shape from last layer
        os = self.layers[-1].shape
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

    def set_optimizer(self, optimizer):
        for layer in self.layers:
            layer.set_optimizer(optimizer)
        return


    def get_minibatch(self, x, y):
        if self.idx + self.batch_size <= self.data_size:
            minibatch_x = x[:, self.idx:self.idx+self.batch_size]
            minibatch_y = y[:, self.idx:self.idx+self.batch_size]
        else:
            # If remaining data is less than size of minibatch, take all remaining data
            minibatch_x = x[:, self.idx:]
            minibatch_y = y[:, self.idx:]

        self.idx += self.batch_size
        return minibatch_x, minibatch_y



########################################################################

    ### Utils ###
    def _check_gradient(self, x, y, layer):
            epsilon = 1e-6
            for i in tqdm(range(layer.shape[0])):
                for j in range(layer.shape[1]):
                    layer.weights[i,j] += epsilon
                    output_plus = self.test_predict(x)
                    loss_plus = self.loss(output_plus, y)
                    layer.weights[i,j] -= 2*epsilon
                    output_minus = self.test_predict(x)
                    loss_minus = self.loss(output_minus, y)
                    raw_gradient = ((loss_plus - loss_minus)/(2*epsilon))
                    gradient = ((loss_plus - loss_minus)/(2*epsilon)).sum()
                    backprop_gradient = layer.gradient[i,j]
                    grad_rel_diff = (gradient-backprop_gradient)/(np.absolute(gradient) + np.absolute(backprop_gradient) + 1)
                    # if not np.isclose(grad_rel_diff, 0.0, rtol=1e-5):
                    if not grad_rel_diff < 1e-5:
                        raise Exception(f"Computed gradient is not correct for layer {layer}")
                    # Reset weights
                    layer.weights[i,j] += epsilon
            if layer.use_bias:
                for i in tqdm(range(len(layer.bias))):
                    layer.bias[i,0] += epsilon
                    output_plus = self.test_predict(x)
                    loss_plus = self.loss(output_plus, y)
                    layer.bias[i,0] -= 2*epsilon
                    output_minus = self.test_predict(x)
                    loss_minus = self.loss(output_minus, y)
                    raw_gradient = ((loss_plus - loss_minus)/(2*epsilon))
                    gradient = ((loss_plus - loss_minus)/(2*epsilon)).sum()
                    backprop_gradient = layer.bias_gradient[i].sum()
                    grad_rel_diff = (gradient-backprop_gradient)/(np.absolute(gradient) + np.absolute(backprop_gradient) + 1)
                    # if not np.isclose(grad_rel_diff, 0.0, rtol=1e-5):
                    if not grad_rel_diff < 1e-5:
                        raise Exception(f"Computed gradient is not correct for bias in layer {layer}")
                    layer.bias[i,0] += epsilon
            logging.info(f'All computed gradients are correct for layer {layer}')