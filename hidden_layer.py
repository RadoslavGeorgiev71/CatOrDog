import numpy as np

import utils as utils

class HiddenLayer:
    """
    Hidden layer of the network
    All parameters have default values specified for the task(cats vs dogs)
    """
    def __init__(self, input_num=3, output_num=4, initialization_strategy=utils.initialize_Xaveir,
                  filter_function=utils.Filter(dim=5, stride=1),
                  activation_function=utils.Sigmoid(),
                  pooling_function=utils.MaxPooling(dim=2, stride=2)):
        self.input_num = input_num
        self.output_num = output_num
        self.weight_dim = filter_function.dim

        weights = []
        biases = []
        # separate filter fucntion for each pair of input and output
        filter_functions = []
        # separate activation and pooling function for each output
        activation_functions = []
        pooling_functions = []
        for i in range(0, output_num):
            weights_per_output = []
            filter_functions_per_output = []
            for j in range(0, input_num):
                weights_per_output.append(initialization_strategy(self.weight_dim, self.weight_dim))
                filter_functions_per_output.append(filter_function)

            weights.append(weights_per_output)
            biases.append(0)

            filter_functions.append(filter_functions_per_output)
            activation_functions.append(activation_function)
            pooling_functions.append(pooling_function)

        self.weights = np.array(weights)
        assert(self.weights.shape == (self.output_num, self.input_num, self.weight_dim, self.weight_dim))
        self.biases = np.array(biases)

        self.filter_functions = filter_functions
        self.activation_functions = activation_functions
        self.pooling_functions = pooling_functions

        self.cache = None
        self.weight_grad = None
        self.bias_grad = None

    def forward(self, matrices):
        """
        Performs a forward pass through the layer
        by computing the filter and then the pooling
        Returns a numpy array of matrices with number
        equal to the desired output number
        """
        # insure we have the right number of square matrices
        assert(matrices.shape[0] == self.input_num)
        assert(matrices.shape[1] == matrices.shape[2])

        # insure the input matrices are large enough
        assert(matrices.shape[1] >= self.weight_dim)

        # save the matrices to be used in backpropagation
        self.cache = matrices

        # calculate the filterer outputs
        filter_results = []
        for i in range(0, self.output_num):
            # sum over the filtered inputs
            filter_sum = None
            for j in range(0, self.input_num):
                filter_result = self.filter_functions[i][j].forward(matrices[j, :, :], self.weights[i, j, :, :])
                # initialize the sum
                if filter_sum is None:
                    filter_sum = np.zeros(filter_result.shape)
                filter_sum += filter_result

            filter_sum += self.biases[i]
            filter_sum = self.activation_functions[i].forward(filter_sum)
            filter_results.append(filter_sum)

        filter_results = np.array(filter_results)
        
        # apply the pooling to the filter results
        pooling_results = []
        for i in range(0, self.output_num):
            filter_result = filter_results[i, :, :]
            pooling_result = self.pooling_functions[i].forward(filter_result)
            pooling_results.append(pooling_result)

        pooling_results = np.array(pooling_results)

        return pooling_results
    
    def backward(self, dupstream):
        """
        Performs a backward pass through the layer.
        """
        assert(dupstream.shape[0] == self.output_num)

        weight_grad = []
        bias_grad = []
        for i in range(0, self.output_num):
            # compute derivatives after filterings
            single_output_dx = self.pooling_functions[i].backward(dupstream[i, : ,:])
            single_output_dx = self.activation_functions[i].backward(single_output_dx)

            # the bias is used in each operation for the output matrix;
            # threfore we sum over the whole matrix
            bias_grad.append(np.sum(single_output_dx))

            weight_grad_output = []
            for j in range(0, self.input_num):
                # compute derivatives befor filtering
                single_input_dx = self.filter_functions[i][j].backward(single_output_dx)
                
                weight_grad_output.append(single_input_dx)

            weight_grad.append(weight_grad_output)

        self.weight_grad = np.array(weight_grad)
        self.bias_grad = np.array(bias_grad)
        assert(self.weights.shape == self.weight_grad.shape)

        # gradient of the input/downstem gradient
        dx = np.sum(self.weight_grad * self.weights, axis=0)
        print(dx.shape)
        return dx