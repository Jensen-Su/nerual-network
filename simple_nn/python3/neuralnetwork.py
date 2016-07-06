"""
neuralnetwork.py
~~~~~~~~~~~~~~~~~~~
A elasitc neural network.

A nerual network has the following components:
    . Layers (ConvLayer, PoolingLayer, FullConnectedLayer, SoftmaxLayer ...)
    . Layer size (type, number of sublayers, number of neurons each layer)
    . Activation function (sigmoid, ReLu, tanh...)

To train a network, we have the following design choices:
    . How to initialize the weights
    . Cost function (quadratic, log(a_y), cross-entropy...)
    . Regularization (L1 norm, L2 norm, dropout, data expansion...)
    . For L1&L2 regularization, the strength \lambda
    . The learning rate \eta
    . For SGD algorithm, the batch size \gamma
    . Even for SGD, there can be many choice (Newton, Newton alike, ...)

The four fundamental equations behind backpropagation:
    http://neuralnetworksanddeeplearning.com/chap2.html
    . delta_L = grad(C, a) * ap(z_L)        ---- The output layer
    . delta_l = dot(w_{l+1}.transpose(), delta_{l+1}) * ap(z_l)
    . grad(C, b_l) = delta_l
    . grad(C, w_l) = dot(delta_l.transpose(), a_{l-1})
    Where:
        . delta is the gradient of cost function C with respective to z
        . z reprensents for the weighed input to a neuron, it has the form:
                            z = w*a + b
        . C is the cost function, a represents for the activation function
        . grad(C, a) is the gradient of C respective to a
        . ap(z_l) is the derivative of a respective to z
        . x * y is element-wise Hadamard product of x and y
        . dot(x, y) is the dot product of x and y
        . L is the number of layers, _L is the last layer
        . _l spcifies certain layer
    
"""
### libraries

# Standard libraries
import sys
import random

# Third-party libraries
import numpy as np
import cost as Cost

class NeuralNetwork(object):
    """
    Neural Network class.
    ~~~~~~~~~~~~~~~~~~~~
    Assemble different layers to form a real neural network.

    The network will be trained in this class using Stochastic 
    gradient descent algorithm.
    """

    def __init__(self, layers, cost = Cost.QuadraticCost()):

        self.layers = layers
        self.cost = cost

    def train(self, training_data, epochs, batch_size, eta,
            lmbda = 0.0, test_data = None, vectorize = False):
        """
        Train the neural network using mini-batch stochastic gradient
        descent algorithm.
        ''training_data'' and ''test_data'' are lists of examples. 
        Each example (x, y) is a tuple of input feture vector x and 
        expected output value (or vector if vectorized) y.
        """
        training_data = list(training_data)
        test_data = list(test_data)

        training_data = training_data[0:10000]
        test_data = test_data[0:5000]

        if training_data: n = len(training_data)
        else: 
            print("Training data not provided. Program exited.\n")
            exit(1)
        
        # begin training...
        for k in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[i : i + batch_size] for i in
                    range(0, n, batch_size)]
            for batch in batches:
                for x, y in batch:
                    # feedforward
                    a = self.feedforward(x)
                    # derivative of Cost w.r.t a
                    Cp_a = self.cost.Cp_a(a, y)
                    # backpropagation
                    Cp_a = self.backprop(Cp_a)
                    # update paremeters
                    self.update(eta, lmbda, batch_size, n)
            
            # evaluate
            if test_data:
                print("Epoch {0}, accurancy on test data: {1} / {2}".format(
                    k, self.evaluate(test_data, vectorize), len(test_data)))
            else:
                print("Epoch {0} complete.".format(k))

    def feedforward(self, a):
        
        for layer in self.layers:
            a = layer.feedforward(a)

        return a

    def backprop(self, Cp_a):
        
        for k in range(len(self.layers)):
            Cp_a = self.layers[-k].backprop(Cp_a)
        return Cp_a

    def update(self, eta, lmbda, batch_size, n):

        for layer in self.layers:
            layer.update(eta, lmbda, batch_size, n)


    def evaluate(self, test_data, vectorize = False):
        """
        Return the number of test inputs for which the neural network
        outputs the correct result. Note that the nerual network's output
        is assumed to be the index of whichever neuron in the final layer
        has the highest activation.
        """
        
        if vectorize:
            test_results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                    for (x, y) in test_data]
        else:
            test_results = [(np.argmax(self.feedforward(x)), y)
                    for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

