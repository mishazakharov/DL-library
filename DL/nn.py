'''
This file contains a few activation functions for my 
deep learning library
I don't know if this realizations is appropriate!
'''

import numpy as np


def sigmoid(input_tensor,derivative=False):
    ''' Sigmoid activation function '''
    if derivative:
        return input_tensor * (1 - input_tensor)
    else:
        return 1 / (1 + np.exp(-input_tensor))
    return None


def tanh(input_tensor,derivative=False):
    ''' Tanh activation function '''
    if derivative:
        return 1 - (np.tanh(input_tensor)) ** 2
    else:
        return np.tanh(input_tensor)
    return None


def relu(input_tensor,derivative=False):
    ''' Relu activation function '''
    if derivative:
        # Relu derivative equals 1 if x > 0 and 0 if x < 0 
        # in 0 technically derivative is undefined
        input_tensor[input_tensor > 0] = 1
        input_tensor[input_tensor <= 0] = 0
        return input_tensor
    else:
        return np.maximum(0,input_tensor)
    return None

def elu(input_tensor,derivative=False,alpha=0.01):
    ''' Elu activation function '''
    if derivative:
        input_tensor[input_tensor > 0] = 1
        input_tensor[input_tensor <= 0] = alpha
    else:
        '''
        Does not work like that!
        input_tensor[input_tensor > 0] = input_tensor
        input_tensor[input_tensor <= 0] = alpha * input_tensor
        '''
    return None

def softmax(input_tensor,derivative=False):
    ''' Softmax activation function '''
    if derivative:
        # I have not realized it yet!
        pass
    else:
        e_x = np.exp(input_tensor - np.max(input_tensor))
        return e_x / e_x.sum(axis=0)
    return None 



