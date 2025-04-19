import numpy as np
import math

import DataParser

def ReLU(x):
    """
    Performs the ReLU activation function
    """

    if x < 0:
        x = 0
    assert(x >= 0)

    return x

def sigmoid(x):
    """
    Performs the sigmoid activation function
    """

    x = 1 / (1 + np.exp(-x))
    assert(x > 0 and x < 1 )

    return x

def intializeHe(dim=5):
    """
    Intialize the weights using He initalization
    most suitable with ReLU activation function
    """
    weights = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)
    assert(weights.shape[0] == weights.shape[1])

    return weights

def maxPooling(segment):
    """
    Performs the max pooling stategy
    by selecting the largest of the values in the segment
    """

    return np.max(segment)

class ConvolutionLayer1:
    def __init__(self, channel_num=3, initialization_stategy=intializeHe,
                  weight_dim=5, activation_function=ReLU, stride_channel=1,
                  pooling_dim=2, pooling_function=maxPooling, stride_pooling=2):
        weights = []
        biases = []
        for i in range(0, channel_num):
            weights.append(initialization_stategy(weight_dim))
            biases.append(0)
        self.weights = np.array(weights)
        self.biases = np.array(biases)

        self.activation_function = activation_function
        self.stride_channel = stride_channel

        self.pooling_dim = pooling_dim
        self.pooling_function = pooling_function
        self.stride_pooling = stride_pooling

    def forward(self, matrix):
        """
        Performs a forward pass through the layer
        by computing all the channels
        """
        # insure we have a square matrix
        assert(matrix.shape[0] == matrix.shape[1])

        # insure the input is large enough
        weight_dim = self.weights.shape[1]
        assert(matrix.shape[0] >= self.weights.shape[1])

        # apply the filter to the channels
        filter_result = []
        for i in range(0, len(self.weights)):
            channel = matrix[:, :, i]
            channel = self.applyFilter(channel, self.weights[i, :, :], self.biases[i])
            filter_result.append(channel)

        filter_result = np.array(filter_result)
        # check that the shape is correct after the filter
        assert(filter_result.shape == (self.weights.shape[0],
                                 math.ceil((matrix.shape[0] - (weight_dim - 1)) / self.stride_channel),
                                 math.ceil((matrix.shape[1] - (weight_dim - 1)) / self.stride_channel)))
        
        # apply the pooling to the channels
        pooling_result = []
        for channel in filter_result:
            channel = self.applyPooling(channel)
            pooling_result.append(channel)

        pooling_result = np.array(pooling_result)
        # check that the shape is correct after the pooling
        assert(pooling_result.shape == (filter_result.shape[0], 
                                 math.ceil((filter_result.shape[1] - (self.pooling_dim - 1)) / self.stride_pooling),
                                 math.ceil((filter_result.shape[2] - (self.pooling_dim - 1)) / self.stride_pooling)))

        return pooling_result
    
    def applyFilter(self, matrix, weights, bias):
        """
        Returns a list of values 
        for one channel of the filter
        """
        weight_dim = weights.shape[0]

        row_values = []
        # iterate over the matrix with the specified stride
        for i in range(0, matrix.shape[0] - (weight_dim - 1), self.stride_channel):
            column_values= []
            for j in range(0, matrix.shape[1] - (weight_dim - 1), self.stride_channel):
                matrix_segment = matrix[i : i + weight_dim, j : j + weight_dim]
                
                # compute the entry of the filter
                result = self.activation_function(np.sum(matrix_segment * weights) + bias) 

                column_values.append(result)

            row_values.append(column_values)

        return row_values
    
    def applyPooling(self, channel):
        """
        Returns a list of pooled values
        for a specified channel
        """
        row_values = []
        # iterate over the channel with the specified stride
        for i in range(0, channel.shape[0] - (self.pooling_dim - 1), self.stride_pooling):
            column_values = []
            for j in range(0, channel.shape[1] - (self.pooling_dim - 1), self.stride_pooling):
                channel_segment = channel[i : i + self.pooling_dim, j : j + self.pooling_dim]

                # compute the entry of the pooling
                result = self.pooling_function(channel_segment)

                column_values.append(result)

            row_values.append(column_values)
        
        return row_values
    
layer1 = ConvolutionLayer1()
data, _ = DataParser.getTrainData()
sample = data[0]
print(sample.shape)
print(layer1.forward(sample))




