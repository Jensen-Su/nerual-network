"""
activation.py
~~~~~~~~~~~~~~
Author: Jensen Su
Date:   2016.07
--------------------
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

class Softmax(Activation):

    @staticmethod
    def func(z):
        """ The functionality. """
        for zi in z:
            if zi > 30:
                print("Some weighted input ''z'' is too large!", sys.stderr)
                exit(1)

        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)
    
    @staticmethod
    def prime(z):
        """ 
        Simply return 1 for the following reason.
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        The correct derivative should be 
        if j == k, da_j / dz_k = a_j * (1 - a_j) > 0;
        if j != k, da_j / dz_k = - a_j ** 2 < 0;

        However here, the derivative is of the output activation w.r.t 
        its weighted input, that is, j == k.

        NOTE: the correct derivative should be
                
                Softmax.func(z) - Softmax.func(z) ** 2.

        But this derivative is not used but be merged to the derivative 
        of ''LogCost''. To make it consistence with the ''LogCost'', we
        simply return 1.
        So that we could also used a FullConnectedLayer as a SoftmaxLayer
        , with ''Softmax'' activation and ''LogCost''. 
        """
        return 1.

