from neurons import Neurons
import numpy as np

# inherit from base class Layer
class Layer(Neurons):

    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, in_neurons=0, out_neurons=0):
        self.weights = np.random.rand(in_neurons, out_neurons) - 0.5
        self.bias = np.random.rand(1, out_neurons) - 0.5
    
    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output
    
    # computes dE/dW, dE/dB for a given output_error = dE/dY. Returns input_error = dE/dX
    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error
        


# inherit from base class layer
class ActivationLayer(Layer):

    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    
    # Returns input_error = dE/dX for a given output_error = dE/dY
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error
        #raise NotImplementedError

