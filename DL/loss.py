'''
This file contains loss functions for
my deep learning library
'''

import numpy as np


class MSE(object):
    ''' MSE error '''
    def loss(self,predicted,actual):
        # Computes a loss function between two tensors!
        return np.sum((predicted - actual)**2)

    def grad(self,predicted,actual):
        # Computes a gradient of loss functions!
        return 2 * (predicted - actual)


class CrossEntropy(object):
    ''' Cross entropy cost function 
    Formula:
            L = -1/n * E(x)[y*log(a) + (1 - y)*log(1-a)], where
            n - number of samples in the training data,
            E(x)[] - sum over x training inputs(attributes),
            y - corresponding desired output
    Derivative:
            D' = y - a
            might be wrong!
    '''

    def loss(self,predicted,actual):
        if actual == 1:
            # Avoiding of log(0) by adding a very small number!
            return -(1/predicted.shape[0]) * np.sum(np.log(predicted + 1e-9))
        else:
            return -(1/predicted.shape[0]) * np.sum(np.log((1 - predicted)+1e-9))

    def grad(self,predicted,actual):
        return predicted - actual
        
