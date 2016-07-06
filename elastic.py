"""
elastic.py
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

#### Network Layers
class BaseLayer(object):
    """
    Layer class: Base class for different type of layers.
    ~~~~~~~~~~~~~~~~~~~~
    Data members: 
    sizes       ---- <type list> sizes of the network
    n_layers    ---- <type int> number of sublayers
    activation  ---- <type Activation> activation function for neurons
    weights     ---- <type list> to store weights
    biases      ---- <type list> to store biases
    neurons     ---- <type list> to store states (outputs) of neurons
    zs          ---- <type list> to store weighted inputs to neurons
    grad_w      ---- <type list> to store gradient of Cost w.r.t weights
    grad_b      ---- <type list> to store gradient of Cost w.r.t biases
    ---------------------
    Methods:
    __init__(self, sizes, activation = Sigmoid())
    size(self)
    model(self)
    feedforward(self, a)
    backprop(self, C_p)
    update(self, eta, lmbda, batch_size, n)
    """
    
    def __init__(self, sizes, activation):
        self.sizes = sizes
        self.n_layers = len(sizes)
        self.activation = activation

    def size(self): return self.sizes

    def model(self): return zip(self.weights, self.biases)

    def feedforward(self, a):
        print("Functionality of ''feedforward'' not provided. Program exited.\n")
        exit(1)
        
    def backprop(self, C_p):
        print("Functionality of ''backprop'' not provided. Program exited.\n")
        exit(1)

    def update(self, eta, lmbda, batch_size, n):
        print("Functionality of ''update'' not provided. Program exited.\n")


class FullConnectedLayer(BaseLayer):
    """
    FullConnectedLayer
    ~~~~~~~~~~~~~~~~~~~~
    Data members: 
    sizes       ---- <type list> sizes of the network
    n_layers    ---- <type int> number of sublayers
    activation  ---- <type Activation> activation function for neurons
    weights     ---- <type list> to store weights
    biases      ---- <type list> to store biases
    neurons     ---- <type list> to store states (outputs) of neurons
    zs          ---- <type list> to store weighted inputs to neurons
    grad_w      ---- <type list> to store gradient of Cost w.r.t weights
    grad_b      ---- <type list> to store gradient of Cost w.r.t biases
    ---------------------
    Methods:
    __init__(self, sizes, activation = Sigmoid())
    size(self)
    model(self)
    feedforward(self, a)
    backprop(self, C_p)
    update(self, eta, lmbda, batch_size, n)
    """

    def __init__(self, sizes, activation = Sigmoid(), normal_initialization = False):
        """
        The list ''sizes'' contains the number of neurons in repective layers
        of the network. For example, sizes = [2, 3, 2] represents 3 layers, with
        the first layer having 2 neurons, the second 3 neurons, and the third 2 
        neurons.

        Note that the input layer may be passed by other layer of another type 
        when connected after the layer, and we don't set biases for this layer.
        Also note that the output layer my be passed to other layer if connected
        before the layer, in this case, just assign the outputs to its inputs.
        For examle, Layer1([3, 2, 4])->Layer2([4, 6, 3])->Layer3([3, 2]). Just
        assign the output of Layer1 to the input Layer2, it will be safe.
        """

        BaseLayer.__init__(self, sizes, activation)

        if normal_initialization:
            self.weights = [np.random.randn(j, i)
                    for i, j in zip(sizes[:-1], sizes[1:])]
        else:
            self.weights = [np.random.randn(j, i) / np.sqrt(i)
                    for i, j in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(j, 1) for j in sizes[1:]]

        self.grad_w = [np.zeros(w.shape) for w in self.weights]
        self.grad_b = [np.zeros(b.shape) for b in self.biases]

    def feedforward(self, a):
        """
        Return output of the network if ''a'' is input.
        """
        self.neurons = [a] # to store activations (outputs) of all layers
        self.zs = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, self.neurons[-1]) + b
            self.zs.append(z)
            self.neurons.append(self.activation.func(z))
        return self.neurons[-1]
    

    def backprop(self, Cp_a):
        """
        Backpropagate the delta error.
        ------------------------------
        Return a tuple whose first component is a list of the gradients of 
        weights and biases, whose second component is the backpropagated delt.
        Cp_a, dC/da: derivative of cost function w.r.t a, output of neurons. 
        """
        # The last layer
        delta = Cp_a * self.activation.prime(self.zs[-1])
        self.grad_b[-1] += delta
        self.grad_w[-1] += np.dot(delta, self.neurons[-2].transpose()) 

        for l in range(2, self.n_layers):
            sp = self.activation.prime(self.zs[-l])  # a.prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp  
            self.grad_b[-l] += delta
            self.grad_w[-l] += np.dot(delta, self.neurons[-l - 1].transpose())

        Cp_a_out = np.dot(self.weights[0].transpose(), delta)

        return Cp_a_out

    def update(self, eta, lmbda, batch_size, n):
        """
        Update the network's weights and biases by applying gradient descent
        algorithm.
        ''eta'' is the learning rate
        ''lmbda'' is the regularization parameter
        ''n'' is the total size of the training data set
        """
        self.weights = [(1 - eta * (lmbda/n)) * w - (eta/batch_size) * delta_w\
                for w, delta_w in zip(self.weights, self.grad_w)]
        self.biases = [ b - (eta / batch_size) * delta_b\
                for b, delta_b in zip(self.biases, self.grad_b)]
        
        # Clear ''grad_w'' and ''grad_b'' so that they are not added to the 
        # next update pass
        for dw, db in zip(self.grad_w, self.grad_b):
            dw.fill(0)
            db.fill(0)


class NeuralNetwork(object):
    """
    Neural Network class.
    ~~~~~~~~~~~~~~~~~~~~
    Assemble different layers to form a real neural network.

    The network will be trained in this class using Stochastic 
    gradient descent algorithm.
    """

    def __init__(self, layers, cost = QuadraticCost()):

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






