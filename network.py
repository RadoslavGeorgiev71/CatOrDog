import numpy as np

import data_parser
import input_layer
import hidden_layer
import output_layer
import utils as utils

class CNN_Network:
    """
    Our convolutional neural network with 3 layers
    """
    def __init__(self, input_layer=input_layer.InputLayer(), hidden_layer=hidden_layer.HiddenLayer(),
                    output_layer=output_layer.OutputLayer(), loss_function=utils.binary_cross_entropy):
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer

        self.loss_function = loss_function

    def forward(self, input):
        """
        Performs a forward pass through the network.
        Returns the predictions.
        """
        input_result = self.input_layer.forward(input)
        hidden_result= self.hidden_layer.forward(input_result)
        output_result = self.output_layer.forward(hidden_result)

        return output_result
    
    def backward(self, dupsteam):
        """
        Performs a backward pass through the network.
        Returns the gradients for the initial input.
        """
        dx = dupsteam
        dx = self.output_layer.backward(dx)
        dx = self.hidden_layer.backward(dx)
        dx = self.input_layer.backward(dx)

        return dx
    
    def optimize_network(self, lr):
        """
        Performs a single optimization of the weights and biases of the layers
        lr: learning rate
        """
        self.input_layer.weights -= lr * self.input_layer.weight_grad
        self.input_layer.biases -= lr * self.input_layer.bias_grad

        self.hidden_layer.weights -= lr * self.hidden_layer.weight_grad
        self.hidden_layer.biases -= lr * self.hidden_layer.bias_grad
        
        self.output_layer.weights -= lr * self.output_layer.weight_grad
        self.output_layer.biases -= lr * self.output_layer.bias_grad

print("Loading data...")
data, _ = data_parser.getTrainData(percentage=0.01)
sample = data[0]

cnn = CNN_Network()
prediction = cnn.forward(sample)
print(prediction)

loss, grad = utils.binary_cross_entropy(prediction, 1)
cnn.backward(grad)

