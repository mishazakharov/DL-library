import numpy as np

class NeuralNetwork(object):
    ''' This class assembles an engine and trains
    our neural network by backpropogating
    '''
    def __init__(self,layers):
        # layers is sequence of layers!
        self.layers = layers

    def forward(self,input_tensor):
        ''' Forward propagation '''
        # for every layer in layers
        for layer in self.layers:
            # Makin forward propagation
            result = layer()
        return result

    def backward(self):
        ''' Backward propagation '''

