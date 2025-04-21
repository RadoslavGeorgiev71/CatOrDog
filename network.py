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


data, _ = data_parser.getTrainData()
sample = data[0]

cnn = CNN_Network()

print(cnn.forward(sample))
