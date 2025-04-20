import numpy as np
import math

import cnn_utils as utils

class OutputLayer:
    """
    The final layer of our CNN.
    """
    def __init__(self, input_size=676, output_size=1,
                  initialization_strtegy=utils.intialize_He, activation_function=utils.sigmoid):
        
        self.weights = initialization_strtegy(input_size, output_size)
        self.biases = np.zeros(output_size)
        
        self.activation_function = activation_function

    def forward(self, input):
        """
        Performs a forward pass through the layer
        Returns a numpy array of output entries
        """
        # flatten the input into a 1d numpy array
        input = input.flatten()

        result = self.activation_function(self.weights.T @ input + self.biases)

        return result