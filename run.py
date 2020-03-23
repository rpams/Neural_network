import numpy as np
from network import Network
from base import *

# training data
xtrain = np.array([[0,0], [0,1], [1,0], [1,1]])
ytrain = np.array([[0], [1], [1], [0]])

# network
net = Network()
net.add(Layer(in_neurons=2, out_neurons=3))
net.add(ActivationLayer(gauss, gauss_prime))
net.add(Layer(in_neurons=3, out_neurons=1))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.loss_parameter(mse, mse_prime)
net.train(xtrain, ytrain, epochs=375, learning_rate=0.1, graphic=True)

# test
net.predict(xtrain)
