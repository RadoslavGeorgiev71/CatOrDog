import numpy as np
import math

import data_parser
import cnn_layer1
import cnn_layer2
import output_layer

class CNN_Network:
    """
    Our convolutional neural network with 3 layers
    """
    def __init__(self, layer1, layer2, output_layer):
        self.layer1 = layer1
        self.layer2 = layer2
        self.output_layer = output_layer

    def forward(self, input):
        """
        Performs a forward pass through the network.
        Returns the predictions.
        """
        output_layer1 = self.layer1.forward(input)
        output_layer2 = self.layer2.forward(output_layer1)
        output = self.output_layer.forward(output_layer2)

        return output

layer1 = cnn_layer1.ConvolutionLayer1()
data, _ = data_parser.getTrainData()
sample = data[0]

layer2 = cnn_layer2.ConvolutionLayer2()


final_layer = output_layer.OutputLayer()

cnn = CNN_Network(layer1, layer2, final_layer)

print(cnn.forward(sample))
