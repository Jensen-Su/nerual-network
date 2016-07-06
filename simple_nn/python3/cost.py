"""
cost.py
~~~~~~~
Define the cost functions.
"""

### libraries

# Standard libraries

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
        Return the error delta from the output layer.
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
        Return the error delta from the output layer. 
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
            if ai < 0:
                print("in CrossEntropyCost.func(a, y)... require a_i > 0, a_i belong to a.")
                exit(1)

        return np.sum(np.nan_to_num(-y * np.log(a) - (1-y) * np.log(1-a)))

    @staticmethod
    def Cp_a(a, y):
        """
        Cp_a, dC/da: the derivative of C w.r.t a
        ''a'' is the output of neurons
        ''y'' is the expected output of neurons
        """
        #return (a - y) # delta
        return (a - y) / (a * (1 - a))

