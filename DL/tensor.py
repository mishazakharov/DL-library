import numpy as np


def tensor(input_data):
    ''' Reshapes input_data in tensor
    M x N x J, where M - number of samples,
    N - number of attributes, J - 1
    So every object is (8,1) matrix!
    '''
    tensor = input_data.reshape(input_data.shape[0],input_data.shape[1],1)
    return tensor


