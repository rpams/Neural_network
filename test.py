from network import Network
from fc_layer import FCLayer
from activation_layer import ActivationLayer
from activations import tanh, tanh_prime
from losses import mse, mse_prime
import numpy as np
"""
weights = np.random.rand(3, 3) - 0.5
biais = np.random.rand(1, 3) - 0.5
input_s = np.arange(9).reshape(3,3)
input_s = np.identity(3)
y = [[0], [1], [0]]

output = np.dot(input_s, weights) + biais
output = tanh(output)
err = mse(y, output)
error = mse_prime(y, output)
error_to_back = tanh_prime(input_s) * error

input_error = np.dot(error_to_back, weights.T)
weights_error = np.dot(input_s.T, error_to_back)
input_error = np.dot(input_error, weights.T)
weights_error = np.dot(input_s.T, input_error)


print("inputs : \n",input_s,"\n")
print("weights : \n",weights,"\n")
print("biais : \n",biais,"\n")
print("y : ",y,"\n")
print("output : \n",output,"\n")
print("Error : \n",y-output,"\n")
print("MSE : ",err,"\n")
print("MSE_prime : \n",error,"\n")
print("ERROR_TOBACK : \n",error_to_back,"\n")
print("Input_error : \n",input_error,"\n")
print("weight_error : \n",weights_error,"\n")"""

x_train = np.array([[[0,0]], 
                    [[0,1]], 
                    [[1,0]], 
                    [[1,1]]])

y_train = np.array([[[0]], 
                    [[1]], 
                    [[1]], 
                    [[0]]])

net = Network()
net.add(FCLayer(2, 2))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(2, 1))
net.add(ActivationLayer(tanh, tanh_prime))

net.use(mse, mse_prime)
net.train(x_train, y_train, epochs=1, learning_rate=0.1)
