"""
run.py
-------
Author: Jensen Su
Date:   2016.07
--------------------
Test Units
------------------------
NOTE: The data and mnist_loader.py are from 
      https://github.com/mnielsen
"""

import mnist_loader
import neuralnetwork as nn
import cost 
import layer 
import activation as A

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

#### sigmoid + quadratic (by default)
#net = nn.NeuralNetwork([layer.FullConnectedLayer([784, 30, 10])])

#### sigmoid + cross-entropy
#net = nn.NeuralNetwork([layer.FullConnectedLayer([784, 30, 10])], \
#        cost.CrossEntropyCost())

#### tanh + quadratic
#net = nn.NeuralNetwork([layer.FullConnectedLayer([784, 30, 10], activation = A.Sigmoid())], \
#        cost.CrossEntropyCost())

#### softmax + log
#net = nn.NeuralNetwork([layer.FullConnectedLayer([784, 30]), 
#    layer.FullConnectedLayer([30, 10], activation = A.Softmax())], cost.LogCost())

#### softmax + log
net = nn.NeuralNetwork([layer.FullConnectedLayer([784, 30]), 
    layer.SoftmaxLayer([30, 10])], cost.LogCost())

##### with full data
#net.train(training_data, 30, 10, .5, lmbda = .5, test_data = test_data)

net.train(training_data, 30, 10, .1, lmbda = 0.0,  test_data = test_data)
