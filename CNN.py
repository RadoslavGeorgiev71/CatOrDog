import numpy as np

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

class ConvolutionLayer1:
    def __init__(self, output_num=2, initialization_stategy=intializeHe, weight_dim=5,
                  activation_function=ReLU, stride=1):
        weights = []
        biases = []
        for i in range(0, output_num):
            weights.append(initialization_stategy(weight_dim))
            biases.append(0)
        self.weights = np.array(weights)
        self.biases = np.array(biases)
        self.activation_function = activation_function
        self.stride = stride

    def forward(self, matrix):
        # insure we have a square matrix
        assert(matrix.shape[0] == matrix.shape[1])

        # insure the input is large enough
        weight_dim = self.weights.shape[1]
        assert(matrix.shape[0] >= self.weights.shape[1])

        result = []
        for weights, bias in zip(self.weights, self.biases):
            row_values = []
            # iterate over the matrix with the specified stride
            for i in range(0, matrix.shape[0] - (weight_dim - 1), self.stride):
                column_values= []
                for j in range(0, matrix.shape[1] - (weight_dim - 1), self.stride):
                    matrix_segment = matrix[i : i + (weight_dim - 1), j : j + (weight_dim - 1)]
                
                    # compute the entry of the layer
                    result = self.activation_function(np.sum(matrix_segment * weights) + bias) 

                    column_values.append(result)

                row_values.append(column_values)

            result.append(row_values)

        result = np.array(result)
        assert(result.shape == (self.weights.shape[0],
                                 matrix.shape[0] - (weight_dim - 1),
                                 matrix.shape[1] - (weight_dim - 1)))
        return result


