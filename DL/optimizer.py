import numpy as np


class SGD(object):
    ''' Stochastic gradient descent '''
    def __init__(self,layer,learning_rate=0.01):
        self.learning_rate = learning_rate
        self.layer = layer

    def step(self):
        '''
        This method takes as input the whole class of current layer
        and takes a step of SGD to update weights ot his layer !
        '''
        self.layer.weights_initializer -= self.learning_rate * self.layer.grad
        



