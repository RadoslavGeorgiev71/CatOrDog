import numpy as np

import utils as utils

class OutputLayer:
    """
    The final layer of our CNN.
    """
    def __init__(self, output_num=1, initialization_strtegy=utils.intialize_He,
                  activation_function=utils.Sigmoid()):
        self.output_num = output_num

        self.weights = []
        self.biases = np.zeros(output_num)
        
        self.initialization_strtegy = initialization_strtegy
        self.activation_function = activation_function

        self.cache = None
        self.cache_shape = None
        self.weight_grad = None
        self.bias_grad = None

    def forward(self, input):
        """
        Performs a forward pass through the layer
        Returns a numpy array of output entries
        """
        self.cache_shape = input.shape

        # flatten the input into a 1d numpy array
        input = input.flatten()
        # cache the input to be used in the backward pass
        self.cache = input.reshape((input.shape[0], 1))

        if not self.weights:
            self.weights = self.initialization_strtegy(input.shape[0], self.output_num)

        z = self.weights.T @ input + self.biases
        result = self.activation_function.forward(z)

        return result
    
    def backward(self, dupstream):
        """
        Performs a backward pass through the layer.
        """
        dx = self.activation_function.backward(dupstream).reshape((self.output_num, 1))

        self.weight_grad = np.dot(self.cache, dx) 
        self.bias_grad = dx

        dx = np.dot(dx, self.weights.T)

        return dx.reshape(self.cache_shape)