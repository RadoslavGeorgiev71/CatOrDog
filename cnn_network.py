import numpy as np
import math

import data_parser
import cnn_layer1
import cnn_layer2

layer1 = cnn_layer1.ConvolutionLayer1()
data, _ = data_parser.getTrainData()
sample = data[0]
print(sample.shape)
output1 = layer1.forward(sample)
print(output1.shape)

layer2 = cnn_layer2.ConvolutionLayer2()
output2 = layer2.forward(output1)
print(output2.shape)