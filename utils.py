import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def intialize_He(in_dim=5, out_dim=5):
    """
    Intialize the weights using He initalization;
    most suitable for ReLU activation function.
    """
    weights = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)

    return weights

def initialize_Xaveir(in_dim=3, out_dim=3):
    """
    Initialize the weights using Xavier initalization;
    most suitable for Sigmoid activation function.
    """
    weights = np.random.randn(in_dim, out_dim) * np.sqrt(1.0 / in_dim)

    return weights

class Filter:
    """
    The class for the filtering of a matrix
    Can perform a forward and a backward pass.
    """
    def __init__(self, dim, stride):
        self.dim = dim
        self.stride = stride

        self.cache = None
        self.weights = None
        self.windows = None

    def forward(self, matrix, weights):
        """
        Perform a filter specified by the weights on the matrix.
        """
        assert(weights.shape == (self.dim, self.dim))

        # save the original matrix and weights to be used in backpropagation
        self.cache = matrix
        self.weights = weights

        # get the dimensions of the input matrix and weights
        matrix_height, matrix_width = matrix.shape
        weights_height, weights_width = weights.shape

        # calculate output dimensions based on stride
        output_height = (matrix_height - weights_height) // self.stride + 1
        output_width = (matrix_width- weights_width) // self.stride + 1
        
        # extract sliding windows from the input matrix
        window_shape = (weights_height, weights_width) 
        windows = sliding_window_view(matrix, window_shape)[::self.stride, ::self.stride]

        # save for backpropagation
        self.windows = windows
        
        # reshape windows in order to perform matrix multiplication
        # shape: (output_height * output_weight, weights_height * weights_width)
        windows_reshaped = windows.reshape(-1, weights_height * weights_width) 
        # shape: (weights_height * weights_width, )
        weights_reshaped = weights.reshape(-1)  

        output = np.dot(windows_reshaped, weights_reshaped)  
        
        # reshape output to match the convolved result dimensions
        output = output.reshape(output_height, output_width)

        return output
    
    def backward(self, dupstream):
        """
        Computes and returns the gradients for the weights.
        """
        assert(dupstream.shape == (self.windows.shape[0], self.windows.shape[1]))

        # multiply each matrix in the windows by the corresponding entry in dupstream
        dx = self.windows * dupstream[..., np.newaxis, np.newaxis]
        dx = np.sum(dx, axis=(0, 1))

         
        stride = 1
        weights_height, weights_width = weights.shape
        matrix_height, matrix_width = a.shape

        # Calculate output dimensions based on stride and filter size
        output_height = (matrix_height - weights_height) // stride + 1
        output_width = (matrix_width - weights_width) // stride + 1

        # Generate indices for the windows in the original matrix (a)
        row_indices = np.arange(output_height)[:, None] * stride  # Shape: (output_height, 1)
        col_indices = np.arange(output_width)[None, :] * stride  # Shape: (1, output_width)

        # Generate grid of row and column indices for each window position
        window_row_indices = row_indices + np.arange(weights_height)  # Shape: (output_height, weights_height)
        window_col_indices = col_indices + np.arange(weights_width)  # Shape: (output_width, weights_width)

        # Broadcast the window indices for the full grid of windows
        window_row_indices = np.tile(window_row_indices, (output_width, 1)).T  # Shape: (output_height, output_width, weights_height)
        window_col_indices = np.tile(window_col_indices, (output_height, 1)).T  # Shape: (output_height, output_width, weights_width)

        # Flatten the window indices for easy indexing
        window_row_indices_flat = window_row_indices.reshape(-1)
        window_col_indices_flat = window_col_indices.reshape(-1)

        # Now, let's compute the index positions within the windows
        # The position in the window is just the relative row/col indices in the filter
        window_positions = np.array(np.meshgrid(np.arange(weights_height), np.arange(weights_width))).T.reshape(-1, 2)

        return dx

class ReLU:
    """
    The class for the ReLU activation function.
    Can perform a forward and backward pass
    """
    def __init__(self):
        # cache to use in the backward pass
        self.cache = None

    def forward(self, x):
        """
        Computes the ReLU activation function.
        """
        y = np.maximum(0, x)
        self.cache = y

        return y
    
    def backward(self, dupstream):
        """
        Computes and returns the derivative of the ReLU function
        """
        dx = dupstream * (self.cache > 0)

        return dx

class Sigmoid:
    """
    The class for the sigmoid activation function.
    Can perform a forward and backward pass
    """
    def __init__(self):
        # cache to use in the backward pass
        self.cache = None

    def forward(self, x):
        """
        Computes the sigmoid activation function.
        """
        x = 1 / (1 + np.exp(-x))
        assert(np.all((x >= 0) & (x <= 1)))

        self.cache = x
        return x

    def backward(self, dupstream):
        """
        Computes and returns the derivative of the sigmoid function
        """
        dx = dupstream * self.cache * (self.cache - 1)

        return dx

class MaxPooling:
    """
    The class for the max pooling function.
    Can perform a forward and a backward pass.
    """
    def __init__(self, dim=2, stride=2):
        self.dim = dim
        self.stride = stride

        self.input_shape = None
        self.pooled_indices = None
    
    def forward(self, matrix):
        """
        Performs max pooling on the given matrix
        """
        self.input_shape = matrix.shape

        # slide through the matrix and compute the maximum value in each window
        window_shape = (self.dim, self.dim)
        windows = sliding_window_view(matrix, window_shape)[:: self.stride, :: self.stride]
        pooled_matrix = windows.max(axis=(-2, -1))
        
        # get the indices of the values choosen in each window
        pooled_indices = windows.reshape(*windows.shape[:-2], -1).argmax(axis=-1)

        out_h, out_w = pooled_matrix.shape
        # indices of the start of each window
        row_indices = np.repeat(np.arange(out_h)[:, None], out_w, axis=1) * self.stride
        col_indices = np.repeat(np.arange(out_w)[None, :], out_h, axis=0) * self.stride

        # offsets based on the pooled indices
        row_offsets = pooled_indices // self.dim  
        col_offsets = pooled_indices % self.dim  

        # Final indices in the original input
        input_rows = row_indices + row_offsets
        input_cols = col_indices + col_offsets

        self.pooled_indices = (input_rows, input_cols)

        return pooled_matrix
    
    def backward(self, dupstream):
        """
        Computes and returns the gradients of the entries
        """
        assert(self.pooled_indices[0].shape == dupstream.shape and
               self.pooled_indices[1].shape == dupstream.shape)

        dx = np.zeros(self.input_shape)

        # add to the gradients where they were chosen in the max pooling
        np.add.at(dx, self.pooled_indices, dupstream)

        return dx


def binary_cross_entropy(prediction, label):
    """
    Computes the loss using binary cross-entropy.
    And its gradient with respect to the prediction.
    Most suitable for binary classfication
    and sigmoid activation function.
    """
    loss = -label * np.log(prediction) - (1 - label) * np.log(1 - prediction)
    grad = -label / prediction + (1 - label) / (1 - prediction)

    return loss, grad