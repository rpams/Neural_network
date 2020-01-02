import numpy as np

# activation functions and their derivative
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return x * (1 - x)

def gauss(x):
   return np.exp(-x)

def gauss_prime(x):
   return -np.exp(-x)
    



def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def Softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def _softmax(z):
    return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

def softmax_prime(z):
    """Computes the gradient of the softmax function.
    z: (T, 1) array of input values where the gradient is computed. T is the
       number of output classes.
    Returns D (T, T) the Jacobian matrix of softmax(z) at the given z. D[i, j]
    is DjSi - the partial derivative of Si w.r.t. input j.
    """
    Sz = softmax(z)
    # -SjSi can be computed using an outer product between Sz and itself. Then
    # we add back Si for the i=j cases by adding a diagonal matrix with the
    # values of Si on its diagonal.
    D = -np.outer(Sz, Sz) + np.diag(Sz.flatten())
    return D