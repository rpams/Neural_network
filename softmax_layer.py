from activation import softmax
class SoftmaxLayer(Neurons):
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = softmax(self.input)
        return self.input
        
    def get_gradients(ytrue, ypred):
       

        return dz
    
    def gradient_descent(W, b, dW, db, learning_rate):
        W = W - learning_rate * dW
        b = b - learning_rate * db
        return W, b


    def backward_propagation(self, output_error, learning_rate):
        da = (-ytrue / ypred)
        matrix = np.matmul(ypred, np.ones((1, 3))) * (np.identity(3) - np.matmul(np.ones((3, 1)), ypred.T))
        dz = np.matmul(matrix, da)