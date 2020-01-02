import numpy as np

# loss functions and derivative 
# mean squared error ou encore moyenne de l'erreur au carré
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)


# getting loss when using logistic regression ( softmax function )

def get_loss(y_true, y_pred):
  return -1 * np.sum(y_true * np.log(y_pred))

def get_loss_numerically_stable(y_true, y_pred):
   return -1 * np.sum(y_true * (y_pred + (-y_pred.max() - np.log(np.sum(np.exp(y_pred-y_pred.max()))))))