# Local
import settings
import activations

# Thirdparty
import numpy as np

class Initializer():

    @staticmethod
    def initialize_weights(initializer, shape):
        if initializer in settings.WEIGHT_INITIALIZERS:
            if initializer == 'uniform':
                weights = np.random.uniform(-1, 1, size=shape)
            elif initializer == 'normal':
                weights = np.random.randn(*shape)
            elif initializer == 'zeros':
                weights = np.zeros(shape)
            elif initializer == 'ones':
                weights = np.ones(shape)
            elif initializer == 'glorot_normal':
                sigma = 2/(shape[0] + shape[1])
                weights = np.random.randn(*shape) * sigma
            elif initializer == 'glorot_uniform':
                limit = np.sqrt(6/(shape[0] + shape[1]))
                weights = np.random.uniform(-limit, +limit, shape)
            elif initializer == 'he_normal':
                sigma = 2/(shape[1])
                weights = np.random.randn(*shape) * sigma
            elif initializer == 'he_uniform':
                limit = np.sqrt(6/(shape[1]))
                weights = np.random.uniform(-limit, +limit, shape)
            return weights
        else:
            raise Exception(f'Initializer {initializer} is not a valid initializer. Please use one of the following: {settings.WEIGHT_INITIALIZERS}')


    @staticmethod
    def initialize_activation(activation_init):
        try:
            activation_classname = ''.join([w.capitalize() for w in activation_init.split()])
            activation = getattr(activations, activation_classname)
            return activation()
        except:
            raise Exception(f'Activation {activation} is not a valid activation.')
