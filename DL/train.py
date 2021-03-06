import numpy as np
import optimizer
import random

class NeuralNetwork(object):
    """ This class assembles an engine and trains
    our neural network by backpropogating

    Args:
        layers(list): sequence of layers in NN
        learning_rate(float): a speed of gradient descent
        loss(object): loss function

    """ 
    def __init__(self,layers,loss,learning_rate=0.01):
        # layers is sequence of layers!
        self.layers = layers
        # Learning rate to SGD
        self.learning_rate = learning_rate
        self.loss = loss

    def forward(self,actual_tensor):
        """ Forward progapation

        Args:
            actual_tensor - input data, like X_train,X_test
            
        """
        result = actual_tensor
        # for every layer in layers
        for layer in self.layers:
            # Makin forward propagation
            result = layer.forward(result) 
        return result

    def backward(self,actual):
        """ Backward propagation

        Args:
            actual(np.ndarray): actual y's, right lables

        """
        # for every layer reversed we backpropagate
        error = None
        # Radnoming i for extracting ith sample from tenosr
        # for stochastic gradient descent
        i = random.randint(0,actual.shape[0]-1)
        for layer in reversed(self.layers):
            error = layer.backward(i,error,actual,self.loss)
            # Updating weights
            optim = optimizer.SGD(layer,self.learning_rate)
            optim.step()
            
    def forward_random(self,new_input_tensor):
        result = new_input_tensor
        for layer in self.layers:
            result = layer.forward_random(result)
        return result

    def train(self,actual_tensor,actual,epochs):
        """ Trains our NN
        """
        for i in range(epochs):
            self.forward(actual_tensor)
            self.backward(actual)
