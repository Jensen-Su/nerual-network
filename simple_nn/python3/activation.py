"""
activation.py
~~~~~~~~~~~~~~
Define the activation funcions
"""

### libraries

# Standard libraries
import random
import sys

# Third-party libraries
import numpy as np

#### Define the activations -----------------------------------------------

class Activation(object):
    """
    Base class for activation function
    """
    @staticmethod
    def func(z):
        """
        The functionality. Need to be implemented by subclass
        """
        print("Activation function is not provided!\n", sys.stderr)
        exit(1)

    @staticmethod
    def prime(z):
        """
        The derivative. Need to be implemented by subclass
        """
        print("Deverivative of activation is not provided!\n", sys.stderr)
        exit(0)

class Sigmoid(Activation):
    
    @staticmethod
    def func(z):
        """ The functionality. """
        return 1. / (1. + np.exp(-z))

    @staticmethod
    def prime(z):
        """ The derivative. """
        return Sigmoid.func(z) * (1. - Sigmoid.func(z))

class Tanh(Activation):

    @staticmethod
    def func(z):
        """ The functionality. """
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

    @staticmethod
    def prime(z):
        """ The derivative. """
        return 1. - Tanh.func(z) ** 2

