"""
cost.py
~~~~~~~
Author: Jensen Su
Date:   2016.07
--------------------
Define the cost functions.
"""

### libraries

# Standard libraries
import sys

# Third-party libraries
import numpy as np

#### Define the cost functions --------------------------------------------

class Cost(object):

    @staticmethod
    def func(a, y):
        """
        The functionality. Need to be implemented by subcalss.
        -------------------------------------------------------
        Return the cost associated with an output ''a'' and desired output
        ''y''.
        """
        print("Cost function not provided. Program exited.\n", sys.stderr)
        exit(1)

    @staticmethod
    def C_p(a, y):
        """
        The error. Need to be implemented by subclass.
        -------------------------------------------------------
        Return the derivative of cost function w.r.t ''a''. 
        """
        print("Delta function not provided. Program exited.\n", sys.stderr)
        exit(1)

class QuadraticCost(Cost):
    
    @staticmethod
    def func(a, y):
        """ 
        Return the cost associated with an output ''a'' and desired output
        ''y''.
        """
        return .5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def Cp_a(a, y):
        """
        Return the derivative of cost function w.r.t ''a''. 
        """
        #return (a - y) * Sigmoid.prime(z)
        return (a - y)

class CrossEntropyCost(Cost):
    
    @staticmethod
    def func(a, y):
        """
        Return the cost associated with an output ''a'' and desired output
        ''y''. 
        Note that np.nan_to_num is used to ensure numerical stability. In
        particular, if both ''a'' and ''y'' have a 1.0 in the same slot, 
        then the expression (1-y) * np.log(1-a) returns nan. The np.nan_to_num
        ensures that that is converted to the correct value(0.0).
        """
        for ai in a:
            if ai < 0 or ai > 1:
                print("in CrossEntropyCost.func(a, y)... require 0 <= a_i <= 1, a_i belong to a.", sys.stderr)
                exit(1)

        return np.sum(np.nan_to_num(-y * np.log(a) - (1-y) * np.log(1-a)))

    @staticmethod
    def Cp_a(a, y):
        """
        Cp_a, dC/da: the derivative of C w.r.t a
        ''a'' is the output of neurons
        ''y'' is the expected output of neurons
        """
        for ai in a:
            if ai < 0 or ai > 1:
                print("in CrossEntropyCost.func(a, y)... require 0 <= a_i <= 1, a_i belong to a.", sys.stderr)
                exit(1)
        #return (a - y) # delta
        return (a - y) / (a * (1 - a))

class LogCost(Cost):
    """
    Only designed for softmax layer.
    --------------------------------
    ''y'' must be vectorized with components in {0, 1}
    For example: y = array([0, ..., 1, 0, ...])
    """

    @staticmethod
    def func(a, y):
        """ Return the total cost. """
        
        return - np.sum(y * np.log(a))

    @staticmethod
    def Cp_a(a, y):
        """
        NOTE: This is the derivative of log cost w.r.t weighted input ''z'',
              rather than activation ''a''.
              This is because dC / dz = (dC / da) * (da / dz), where
              dC / da = - y * 1.0 / a, da / dz = a - a ** 2 for softmax activation.
              That make dC / dz = a - y. Since some ''a_i'' in ''a'' may be zero,
              seperating dC / da and da / dz will make the error ''divided by zero'' 
            
              This is to make it consistence with the class ''Softmax'' in 
              ''activation.py'', such that we could also used a FullConnectedLayer 
              with ''Softmax'' activation and ''LogCost'' as a SoftmaxLayer.
        """
        # return - y * 1. / a
        return a - y
        
