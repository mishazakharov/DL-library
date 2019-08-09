import numpy as np


class SGD(object):
    """ This class realizes stochastic gradient descent

    The only difference between a simple gradient descent is that
    this one uses only one sample for each iteration

    Args:
        layer(object): this objects stands for a layer we optimizing
        learning_rate(float): a constant used in that optimization

    """
    def __init__(self,layer,learning_rate=0.01):
        self.learning_rate = learning_rate
        self.layer = layer

    def step(self):
        '''
        This method takes as input the whole class of current layer
        and takes a step of SGD to update weights ot his layer !
        '''
        self.layer.weights_initializer -= self.learning_rate * self.layer.grad
        

class AdamOptimizer(object):
    """ This class realizes Adam Optimizer
    """
    def __init__(object):
        # Soon...
        raise NotImplementedError


