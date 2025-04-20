import numpy as np

# funcions given as parameters to the network
def ReLU(x):
    """
    Performs the ReLU activation function.
    """

    if x < 0:
        x = 0
    assert(x >= 0)

    return x

def sigmoid(x):
    """
    Performs the sigmoid activation function.
    """

    x = 1 / (1 + np.exp(-x))
    assert(x > 0 and x < 1 )

    return x

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

def max_pooling(segment):
    """
    Performs the max pooling stategy
    by selecting the largest of the values in the segment.
    """

    return np.max(segment)