import numpy as np

# funcions given as parameters to the network
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
    Intialize the weights using He initalization;
    most suitable for ReLU activation function
    """
    weights = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)
    assert(weights.shape[0] == weights.shape[1])

    return weights

def initializeXaveir(dim=3):
    """
    Initialize the weights using Xavier initalization;
    most suitable for Sigmoid activation function
    """
    weights = np.random.randn(dim, dim) * np.sqrt(1.0 / dim)
    assert(weights.shape[0] == weights.shape[1])

    return weights

def maxPooling(segment):
    """
    Performs the max pooling stategy
    by selecting the largest of the values in the segment
    """

    return np.max(segment)