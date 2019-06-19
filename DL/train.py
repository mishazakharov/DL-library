import numpy as np
import optimizer
import random

class NeuralNetwork(object):
    ''' This class assembles an engine and trains
    our neural network by backpropogating
    '''
    def __init__(self,layers,actual,learning_rate):
        # layers is sequence of layers!
        self.layers = layers
        # Actual answers
        self.actual = actual
        # Learning rate to SGD
        self.learning_rate = learning_rate

    def forward(self):
        ''' Forward propagation '''
        # for every layer in layers
        for layer in self.layers:
            # Makin forward propagation
            result = layer.forward() 
        return result

    def backward(self):
        ''' Backward propagation '''
        # for every layer reversed we backpropagate
        error = None
        # Radnoming i for extracting ith sample from tenosr
        # for stochastic gradient descent
        i = random.randint(0,self.actual.shape[0]-1)
        for layer in reversed(self.layers):
            error = layer.backward(i,error,self.actual)
            # Updating weights
            optim = optimizer.SGD(layer,self.learning_rate)
            optim.step()
            
    def forward_random(self,new_input_tensor):
        result = new_input_tensor
        for layer in self.layers:
            result = layer.forward_random(result)
        return result
