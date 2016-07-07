"""
neuralnetwork.py
~~~~~~~~~~~~~~~~~~~
Author: Jensen Su
Date:   2016.07
--------------------
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

        if type(layers) != list:
            print("Illegal input, layers must be a list.", sys.stderr)
            exit(1)

        for l in range(1, len(layers)):
            sizes0 = layers[l - 1].size()
            sizes1 = layers[l].size()
            if sizes0[-1] != sizes1[0]:
                print("Illegal input layer sizes.", sys.stderr)
                exit(1)

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
        NOTE:
        ''y'' in training_data is vectorized, while
        ''y'' in test_data is not.
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
            Cp_a = self.layers[-k - 1].backprop(Cp_a)
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

