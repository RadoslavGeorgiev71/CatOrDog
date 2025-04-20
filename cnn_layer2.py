import numpy as np
import math

import cnn_utils as utils

class ConvolutionLayer2:
    """
    The second layer of our network
    All parameters have default values specified for the task(cats vs dogs)
    """
    def __init__(self, channel_num=3, output_num=4, initialization_strategy=utils.initialize_Xaveir,
                  weight_dim=5, activation_function=utils.sigmoid, stride_channel=1,
                  pooling_dim=2, pooling_function=utils.max_pooling, stride_pooling=2):
        # weights have dimention output_num x channel_num x weight_dim x weight_dim
        weights = []
        biases = []
        for i in range(0, output_num):
            weight_channels = []
            for j in range(0, channel_num):
                weight_channels.append(initialization_strategy(weight_dim, weight_dim))

            weights.append(weight_channels)
            biases.append(0)
        self.weights = np.array(weights)
        assert(self.weights.shape == (output_num, channel_num, weight_dim, weight_dim))
        self.biases = np.array(biases)

        self.activation_function = activation_function
        self.stride_channel = stride_channel

        self.pooling_dim = pooling_dim
        self.pooling_function = pooling_function
        self.stride_pooling = stride_pooling

    def forward(self, matrices):
        """
        Performs a forward pass through the layer
        by computing the filter and then the pooling
        Returns a numpy array of matrices with number
        equal to the desired output number
        """
        # insure we have the right number of square matrices
        assert(matrices.shape[0] == self.weights.shape[1])
        assert(matrices.shape[1] == matrices.shape[2])

        # insure the input matrices are large enough
        assert(matrices.shape[1] >= self.weights.shape[2])

        filter_result = []
        for i in range(0, self.weights.shape[0]):
            single_output = self.applyFilter(matrices, self.weights[i], self.biases[i])
            filter_result.append(single_output)

        filter_result = np.array(filter_result)
        # check that the shape is correct after the filter
        weight_dim = self.weights.shape[2]
        assert(filter_result.shape == (self.weights.shape[0],
                                 math.ceil((matrices.shape[1] - (weight_dim - 1)) / self.stride_channel),
                                 math.ceil((matrices.shape[2] - (weight_dim - 1)) / self.stride_channel)))
        
        # apply the pooling to the filter results
        pooling_result = []
        for filter_output in filter_result:
            pooling_output = self.applyPooling(filter_output)
            pooling_result.append(pooling_output)

        pooling_result = np.array(pooling_result)
        # check that the shape is correct after the pooling
        assert(pooling_result.shape == (filter_result.shape[0], 
                                 math.ceil((filter_result.shape[1] - (self.pooling_dim - 1)) / self.stride_pooling),
                                 math.ceil((filter_result.shape[2] - (self.pooling_dim - 1)) / self.stride_pooling)))

        return pooling_result

    def applyFilter(self, matrices, weights, bias):
        """
        Returns a list of values 
        for one output
        """
        weight_dim = weights.shape[1]

        row_values = []
        # iterate over the matrices with the specified stride
        for i in range(0, matrices.shape[1] - (weight_dim - 1), self.stride_channel):
            column_values= []
            for j in range(0, matrices.shape[2] - (weight_dim - 1), self.stride_channel):
                matrix_segments = matrices[:, i : i + weight_dim, j : j + weight_dim]

                # compute the entry of the filter
                result = self.activation_function(sum(
                    [np.sum(matrix_segments[k] * self.weights[k]) for k in range(0, matrix_segments.shape[0])]
                    ) + bias)
                
                column_values.append(result)

            row_values.append(column_values)

        return row_values
    
    def applyPooling(self, input):
        """
        Returns a list of pooled values
        for an input matrix
        """

        row_values = []
        # iterate over the channel with the specified stride
        for i in range(0, input.shape[0] - (self.pooling_dim - 1), self.stride_pooling):
            column_values = []
            for j in range(0, input.shape[1] - (self.pooling_dim - 1), self.stride_pooling):
                input_segment = input[i : i + self.pooling_dim, j : j + self.pooling_dim]

                # compute the entry of the pooling
                result = self.pooling_function(input_segment)

                column_values.append(result)

            row_values.append(column_values)
        
        return row_values