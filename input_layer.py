import numpy as np
import math

import utils as utils

class InputLayer:
    """
    The fist layer of our network
    All parameters have default values specified for the task(cats vs dogs)
    """
    def __init__(self, channel_num=3, initialization_stategy=utils.intialize_He,
                  filter_function=utils.Filter(dim=5, stride=1),
                  activation_function=utils.ReLU(),
                  pooling_function=utils.MaxPooling(dim=2, stride=2)):
        self.channel_num = channel_num
        self.weight_dim = filter_function.dim
        
        weights = []
        biases = []
        # separate functions for each channel(easier for backpropagation)
        filter_functions = []
        activation_functions = []
        pooling_functions = []
        for i in range(0, channel_num):
            weights.append(initialization_stategy(self.weight_dim, self.weight_dim))
            biases.append(0)

            filter_functions.append(filter_function)
            activation_functions.append(activation_function)
            pooling_functions.append(pooling_function)

        self.weights = np.array(weights)
        self.biases = np.array(biases)

        self.filter_functions = filter_functions
        self.activation_functions = activation_functions
        self.pooling_functions = pooling_functions

        self.cache = None
        self.weight_grad = None
        self.bias_grad = None

    def forward(self, matrix):
        """
        Performs a forward pass through the layer
        by computing the filter and then the pooling for all channels
        Returns a numpy array of matrices for each channel
        """
        # insure we have a square matrix
        assert(matrix.shape[0] == matrix.shape[1])

        # insure the input is large enough
        assert(matrix.shape[0] >= self.weight_dim)

        self.cache = matrix

        # apply the filter to the channels
        filter_result = []
        for i in range(0, self.channel_num):
            channel = matrix[:, :, i]
            channel = self.filter_functions[i].forward(channel, self.weights[i, :, :])
            channel = self.activation_functions[i].forward(channel + self.biases[i])
            filter_result.append(channel)

        filter_result = np.array(filter_result)
        
        # apply the pooling to the channels
        pooling_result = []
        for channel in filter_result:
            channel = self.pooling_functions[i].forward(channel)
            pooling_result.append(channel)

        pooling_result = np.array(pooling_result)

        return pooling_result
    
    def backward(self, dupstream):
        """
        Performs a backward pass through the layer.
        """
        assert(dupstream.shape[0] == self.channel_num)

        weight_grad = []
        bias_grad = []
        for i in range(0, self.channel_num):
            single_input_dx = self.pooling_functions[i].backward(dupstream[i, :, :])
            single_input_dx = self.activation_functions[i].backward(single_input_dx)
            single_input_dx = self.filter_functions[i].backward(single_input_dx)

            weight_grad.append(single_input_dx)
            bias_grad.append(np.sum(single_input_dx))

        self.weight_grad = np.array(weight_grad)
        self.bias_grad = np.array(bias_grad)