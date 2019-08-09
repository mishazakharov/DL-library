import numpy as np
import nn
import loss
import math

class Dense(object):
    """ This class builds a single dense layer
    
    Dense layer is basically just an object that computes 
    weighted summ of inputs and passes it through a activational
    function, which you can choose

    Args:
        input_tensor(np.ndarray): input data/training data
        units(int): number of neurons in a layer
        activation_function(object): activation function used in a layer
        weights_initializer(str): type of weights initializer

    """
    def __init__(self,input_tensor,units,activation_function='',
                                        weights_initializer=''):
        ''' As input __init__ receives input_tensor(M x N x 1),
        so every object in input_data is (n,1) matrix
        Default initialization - Xavier weights initialization!
        '''
        # Creating weights matrix for current layer
        # m x n , where m - number of units in the previous layer and
        # n - number of units in the current layer
        self.input_tensor = input_tensor
        self.units = units
        self.weights_initializer = weights_initializer
        self.bias = np.ones((units,1))
        self.activation_function = activation_function
        if self.weights_initializer == 'he':
            pass
        # Formula of Xavier initialization!
        xavier = math.sqrt(2 / (self.input_tensor.shape[0] + self.units))
        # With xavier initalization needed more epochs to converge!
        self.weights_initializer = np.full((units,self.input_tensor.shape[1]),xavier)
        # Place to hold our gradients for backpropagation
        self.grad = np.empty(self.weights_initializer.shape) 
        # Linear output(before passing it through activation function!)
        self.output = np.empty((self.input_tensor.shape[0],units,1)) 
        self.activation_output = np.empty((self.input_tensor.shape[0],
                                                            units,1))

    def __call__(self):
        ''' Special method to return every single time object gets called.
        I am NOT sure that this realization is the best one. Might change it 
        later!!!!
        If we have layer with some activation function this method returns
        activation output
        Else it just returns linear output
        Both are in self.output and self.activation_output!
        '''
        if self.activation_function == '':
            return self.output
        else:
            return self.activation_output
        return None
   
    def forward(self,actual_tensor):
        ''' Computes output corresponding to the input
        tensor. output = wx + b
        Computes for the whole passed dataset not just for a single
        object, so later we have to manually extracr object from tensors!
        Actual_tensor - input tensor, self.input_tensor is used only to create
        matrixes in right dimensions!
        '''
        # So i can get this variable in backprop method!
        self.actual_tensor = actual_tensor
        for i,sample in enumerate(self.actual_tensor):
            # Here we already have (n,1) matrix in sample variable!
            # Iterate through weight matrix
            new = []
            for b,row in enumerate(self.weights_initializer):
                result = np.dot(sample.T,row)
                new.append(result)
            self.output[i] = np.array(new).reshape(-1,1)
        # Reshaping output in the right form. it must be a tensor!
        self.output = self.output.reshape(self.input_tensor.shape[0],self.units,1)
        # Here our tensor have already passed weighted summation
        # Now we need to pass it through activation function
        if self.activation_function == '':
            # If we dont have any activation function
            # then activation output is just linear output
            self.activation_output = self.output
        else:
            # If we have activation function passed to this class
            # then we just pass tensor through activation function
            # and get our activation output
            self.activation_output = self.activation_function(self.output)

        # Rewrote __call__ method just to make everything cleaner...
        if self.activation_function == '':
            return self.output
        else:
            return self.activation_output
        return None

    def backward(self,i,gradient=None,actual=None,loss=None):
        """
        --
        [i] means that i am extracting ith object from tensor due to 
        specifics of this realization!
        --
        Does backward propagation!| For the last layer ->
        Takes as an input derivative of loss with respect to its outputs
        For a hidden layer it takse as an input a derivative of loss function
        with respect to inputs of the next layer(reversed previous!)
        Formulas:
                dD/dy = MSE.grad()
                dD/dw = dD/dy * dy/dw (chain rule),where
                y = f(x0 + x1w1 + x2w2 + ...) so 
                dy/dw = f'(S) * x, where S is x0 + x1w1 + x2w2 + ...
                dD/dx = dD/dy * dy/dx (chain rule), where
                x - is an input and y - is and ouput, w - weights!
                dy/dx = f'(S) * w
        If this method doesn't recieve anything -> grad=None, then
        this layer is the last one and it computes MSE on outputs
        Else it just receives grads and follows the algorithm!

        Variables:
                errors = gradient of L with respect to outputs
                self.grad = gradient of L with respect to weights
                gradient_inputs = gradient of L with respect to inputs

        """
        error = gradient
        # If this layer is the last layer in neural network:
        if not isinstance(error,np.ndarray):
            loss_function = loss
            if isinstance(self.activation_function,int):
                pass
            # Gradient of loss function with respect to its output
            error = loss_function.grad(self.activation_output[i],actual[i])
        # Now we need to calculate gradient of loss function with respect
        # to the weights and store it in self.grad and use it in
        # GRADIENT DESCENT TO UPDATE WEIGHTS OF THIS LAYER!!!!!
        #print(error.shape,'Error shape!')
        #print(error,'VALUE OF ERROR')
        #print(self.output[i],'THIS IS LINEAR OUTPUT ON iTH object')
        #print(self.activation_function(self.output[i],derivative=True).shape,'This is AFOD')
        #print(self.activation_function(self.output[i],derivative=True),'VALUE OF AFOD')
        #print(self.actual_tensor[i].shape,'This is the only right one TENSOR!@')
        #print(self.actual_tensor[i],'VALUE OF ACTUAL TENSOR')
        self.grad = ((error.T *
                self.activation_function(self.output[i],derivative=True)) * 
                self.actual_tensor[i].T)
        #print(self.grad.shape,'THIS IS GRADIENT!')
        #print(self.grad,'VALUES of gradients')
        # Now we need to calculate gradient of loss function with respect
        # to inputs and pass it to previous layer as a gradient of loss 
        # function with respect to its outputs!
        gradients_inputs = np.dot((error *
                self.activation_function(self.output[i],derivative=True).T),
                self.weights_initializer)
        #print(gradients_inputs.shape,'GRADIENT INPUTS!')
        #print(gradients_inputs,'VALUES of gradient inputs')
        # Passing it to previous layer
        return gradients_inputs
        
    def forward_random(self,new_input_tensor):
        """
        Does forward propagation on a new tensor
        Will change it later
        """
        new_result = np.empty((new_input_tensor.shape[0],self.units,1))
        for i,sample in enumerate(new_input_tensor):
            new = []
            for b,row in enumerate(self.weights_initializer):
                result = np.dot(sample.T,row)
                new.append(result)
            new_result[i] = np.array(new).reshape(-1,1)
        new_result = new_result.reshape(new_input_tensor.shape[0],self.units,1)
        if self.activation_function == '':
            new_result_activation = new_result
        else:
            new_result_activation = self.activation_function(new_result)

        if self.activation_function == '':
            return new_result
        else:
            return new_result_activation
        return None

