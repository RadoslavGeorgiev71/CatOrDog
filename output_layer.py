import numpy as np

import utils as utils

class OutputLayer:
    """
    The final layer of our CNN.
    """
    def __init__(self, output_size=1, initialization_strtegy=utils.intialize_He,
                  activation_function=utils.Sigmoid()):
        self.output_size = output_size

        self.weights = []
        self.biases = np.zeros(output_size)
        
        self.initialization_strtegy = initialization_strtegy
        self.activation_function = activation_function

        self.cache = None
        self.weight_grad = None
        self.bias_grad = None

    def forward(self, input):
        """
        Performs a forward pass through the layer
        Returns a numpy array of output entries
        """
        # flatten the input into a 1d numpy array
        input = input.flatten()
        # cache the input to be used in the backward pass
        self.cache = input

        if not self.weights:
            self.weights = self.initialization_strtegy(input.shape[0], self.output_size)

        z = self.weights.T @ input + self.biases
        result = self.activation_function.forward(z)

        return result
    
    def backward(self, dupstream):
        """
        Performs a backward pass through the layer.
        """
        dx = self.activation_function.backward(dupstream)

        self.weight_grad = np.dot(self.cache.T, dx) 
        self.bias_grad = dx

        dx = np.dot(dx, self.weights.T)

        return dx.reshape(self.cache.shape)