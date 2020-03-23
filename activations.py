import numpy as np

# activation functions and their derivative

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

def sigmoid(x):
    return 1 / 1 + np.exp(-x)

def sigmoid_prime(x):
    return x * (1 - x)

def gauss(x):
   return np.exp(-x)

def gauss_prime(x):
   return -np.exp(-x)