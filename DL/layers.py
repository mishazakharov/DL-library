import numpy as np

class Dense(object):
    ''' Dense layer. It just computes weighted summ of inputs
        I have to realize multi-sample inputs
    '''
    def __init__(self,input_tensor,units,weights_initializer=''):
        ''' As input __init__ receives input_tensor(M x N x 1),
        so every object in input_data is (n,1) matrix
        '''
        # Creating weights matrix for current layer
        # m x n , where m - number of units in the previous layer and
        # n - number of units in the current layer
        self.input_tensor = input_tensor
        self.units = units
        self.weights_initializer = weights_initializer
        if self.weights_initializer == 'he':
            pass
        self.weights_initializer = np.ones((units,self.input_tensor.shape[1]))
        # Place to hold our gradients for backpropagation
        self.grad = dict() 
        # Ouputs
        self.output = np.empty((self.input_tensor.shape[0],units,1)) 
        # Returns weighted summ of inputs
        for i,sample in enumerate(self.input_tensor):
            # Here we already have (n,1) matrix in sample variable!
            # Iterarate thrupugh weight matrix
            new = []
            for b,row in enumerate(self.weights_initializer):
                result = np.dot(sample.T,row)
                self.output[i,b] = result
                #new.append(result)
            #self.output[i] = np.array(new).reshape(-1,1)
        self.output = self.output.reshape(self.input_tensor.shape[0],
                                    units,1)

    def __call__(self):
        ''' Special method to return every single time object gets called.
        I am NOT sure that this realization is the best one. Might change it 
        later!!!!
        '''
        return self.output 
   
        
