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