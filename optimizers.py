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

    def update_weights(self, weights, learning_rate, batch_size, gradient):
        weights -= learning_rate/batch_size * gradient
        return weights

    def update_bias(self, bias, learning_rate, batch_size, bias_gradient):
        bias -= learning_rate/batch_size * bias_gradient
        return bias