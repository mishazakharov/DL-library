'''
This file contains a few activation functions for my 
deep learning library
I don't know if this realizations is appropriate!
'''

import numpy as np

def linear(input_tensor,derivative=False):
    ''' Linear activation function 
    If we have linear activation function on a layer
    partial derivative on activation function with respect
    to a linear output turns to be ONE.
    That's why derivative equals 1
    Formulas:
            dD/dw = dD/dy * dy/dw = 2 * (y - a) * input_of_layer
    so in this formula there is no partial derivative of activation function!
    '''
    if derivative:
        return np.array([1])
    else:
        return input_tensor
    return None


def sigmoid(input_tensor,derivative=False):
    ''' Sigmoid activation function 
    Formulas:
            s(x) = 1 / 1 + exp(-x)
            s'(x) = s(x) * (1 - s(x))
    '''
    if derivative:
        return sigmoid(input_tensor) * (1 - sigmoid(input_tensor))
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
    new_tensor = input_tensor.copy()
    if derivative:
        # Relu derivative equals 1 if x > 0 and 0 if x < 0 
        # in 0 technically derivative is undefined
        new_tensor[new_tensor > 0] = 1
        new_tensor[new_tensor <= 0] = 0
        return new_tensor 
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
    ''' Softmax activation function 
    Info: https://ru.wikipedia.org/wiki/Softmax
    '''
    if derivative:
        # I am not sure about this realization!
        result = (softmax(inpu_tensor,derivative=False) * 
                        (1 - softmax(input_tensor,derivative=False)))
        return result
    else:
        # Numerically more stable version of softmax activation function is
        # not just np.exp(input_tensor) but this...
        e_x = np.exp(input_tensor - np.max(input_tensor))
        return e_x / np.sum(e_x) 
    return None 



