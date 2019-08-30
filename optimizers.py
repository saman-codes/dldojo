# Thirdparty
import numpy as np

class Optimizer():

    def __init__(self, **kwargs):
        return

    def update_weights(self, *args):
        return

    def update_bias(self, *args):
        return

class MinibatchSgd(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def update_weights(self, weights, lr, batch_size, gradient):
        weights -= lr/batch_size * gradient
        return weights

    def update_bias(self, bias, lr, batch_size, bias_gradient):
        bias -= lr/batch_size * bias_gradient
        return bias

class Momentum(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.velocity = 0
        self.bias_velocity = 0
        return

    def update_weights(self, weights, lr, batch_size, gradient, mu=9e-1):
        self.velocity = mu * self.velocity - lr/batch_size * gradient
        weights += self.velocity
        return weights

    def update_bias(self, bias, lr, batch_size, bias_gradient, mu=9e-1):
        self.bias_velocity = mu * self.bias_velocity - lr/batch_size * bias_gradient
        bias += self.bias_velocity
        return bias

class NesterovMomentum(Optimizer):
    '''
    Note: this implementation uses an approximation of the
    Nesterov Momentum formula which is valid only for large
    values of mu, see: stackoverflow.com/questions/50774683
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.velocity = 0
        self.previous_velocity = 0
        self.bias_velocity = 0
        self.previous_bias_velocity = 0
        return

    def update_weights(self, weights, lr, batch_size, gradient, mu=9e-1):
        self.previous_velocity = self.velocity
        self.velocity = mu * self.velocity - lr/batch_size * gradient
        weights += -mu * self.previous_velocity + (1 + mu) * self.velocity
        return weights

    def update_bias(self, bias, lr, batch_size, bias_gradient, mu=9e-1):
        self.previous_bias_velocity = self.bias_velocity
        self.bias_velocity = mu * self.bias_velocity - lr/batch_size * bias_gradient
        bias += -mu * self.previous_bias_velocity + (1 + mu) * self.bias_velocity
        return bias

class Adagrad(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cache = 0
        self.bias_cache = 0
        return

    def update_weights(self, weights, lr, batch_size, gradient, eps=1e-6):
        self.cache += gradient ** 2
        weights -= lr / (np.sqrt(self.cache) + eps) * gradient
        return weights

    def update_bias(self, bias, lr, batch_size, bias_gradient, eps=1e-6):
        self.bias_cache += bias_gradient ** 2
        bias -= lr / (np.sqrt(self.bias_cache) + eps) * bias_gradient
        return bias

class Rmsprop(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cache = 0
        self.bias_cache = 0
        return

    def update_weights(self, weights, lr, batch_size, gradient, eps=1e-6, decay_rate=0.99):
        self.cache += decay_rate * self.cache + (1 - decay_rate) * gradient ** 2
        weights -= lr / (np.sqrt(self.cache) + eps) * gradient
        return weights

    def update_bias(self, bias, lr, batch_size, bias_gradient, eps=1e-6, decay_rate=0.99):
        self.bias_cache += decay_rate * self.bias_cache + (1 - decay_rate) * bias_gradient ** 2
        bias -= lr / (np.sqrt(self.bias_cache) + eps) * bias_gradient
        return bias

class Adam(Optimizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.t = 0
        self.cache_m = 0
        self.cache_v = 0
        self.bias_cache_m = 0
        self.bias_cache_v = 0
        return

    def update_weights(self, weights, lr, batch_size, gradient, eps=1e-8, beta1=0.9, beta2=0.999):
        self.t += 1
        self.cache_m = beta1*self.cache_m + (1-beta1)*gradient
        # Bias correction
        m_corrected = self.cache_m / (1-beta1**self.t)
        self.cache_v = beta2 * self.cache_v + (1-beta2)*(gradient**2)
        v_corrected = self.cache_v / (1-beta2**self.t)
        weights -= lr * m_corrected / (np.sqrt(v_corrected) + eps)
        return weights

    def update_bias(self, bias, lr, batch_size, bias_gradient, eps=1e-8, beta1=0.9, beta2=0.999):
        self.bias_cache_m = beta1*self.bias_cache_m + (1-beta1)*bias_gradient
        # Bias correction
        m_corrected = self.bias_cache_m / (1-beta1**self.t)
        self.bias_cache_v = beta2 * self.bias_cache_v + (1-beta2)*(bias_gradient**2)
        v_corrected = self.bias_cache_v / (1-beta2**self.t)
        bias -= lr * m_corrected / (np.sqrt(v_corrected) + eps)
        return bias