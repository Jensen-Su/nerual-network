"""
run.py
-------
Author: Jensen Su
Date:   2016.07
--------------------
Test Units
------------------------
"""

import mnist_loader
import neuralnetwork as nn
import cost 
import layer 
import activation as A

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

#net = nn.NeuralNetwork([layer.FullConnectedLayer([784, 30, 10])], cost.CrossEntropyCost())
#net = nn.NeuralNetwork([layer.FullConnectedLayer([784, 30]), 
#    layer.FullConnectedLayer([30, 10], activation = A.Softmax())], cost.LogCost())

net = nn.NeuralNetwork([layer.FullConnectedLayer([784, 30]), 
    layer.SoftmaxLayer([30, 10])], cost.LogCost())
#92.64%
#net.train(training_data, 30, 10, .5, lmbda = .5, test_data = test_data)
net.train(training_data, 30, 10, .1, lmbda = 0.0,  test_data = test_data)
