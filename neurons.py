
# Base class
class Neurons:

    def __init__(self):
        self.input = None
        self.output = None


    # Computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # Computes dE/dX for a given dE/dY ( and update parameters if any )
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError
