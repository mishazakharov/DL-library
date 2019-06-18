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
