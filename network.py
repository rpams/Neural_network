import numpy as np

class Network:
    def __init__(self, softmax=False):
        self.layers = []
        self.loss = None
        self.loss_prime = None
        self.softmax = softmax
        
        # for train report
        self.errors = []
        self.epoch = []
        self.accuracy = 0

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def loss_parameter(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        
        print("\n________________________________________ ")
        print("\nPredictions : ")
        for i, elt in enumerate(result):
            if elt[0][0] >= 0.5:
                elt[0][0] = 1
            else:
                elt[0][0] = 0
            print(i+1,"=>",elt)
        print("min error = ", min(self.errors),"\n")
        print("________________________________________ END")
    
    # Organize data
    def struct_train(self, xtrain, ytrain):
        xtrain = [np.array([elt]) for elt in xtrain]
        ytrain = [np.array([elt]) for elt in ytrain]
        return xtrain, ytrain


    # train the network
    def train(self, x_train, y_train, epochs=0, learning_rate=0, graphic=False, name='gradient_descent'):
        # sample dimension first
        samples = len(x_train)
        x_train, y_train = self.struct_train(x_train, y_train) 
        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch(s) %d/%d => err = %.10f' % (i+1, epochs, err))
            #print(f'epoch {i+1}/epochs => error={err}')
            
            # setting values of training report
            self.epoch.append(i)
            self.errors.append(err)

        if graphic:
            self.get_train_graph(name)
            
            
            
    def final_state(self):
        # sending training report
        return (self.errors, self.epoch)

    
    def get_train_graph(self, name='gradient_descent'):
        errors, iterations = self.final_state()
        import matplotlib.pyplot as plt
        plt.grid(True)
        plt.title("Gradient Descent => min = "+str(errors[-1]))
        plt.plot(iterations, errors, label="Gradient")
        #plt.annotate(errors[-1], (iterations[-1], errors[-1]))
        #plt.text(iterations[-1], errors[-1], r'min')
        plt.xlabel('iterations')
        plt.ylabel('error')
        plt.savefig(name+'.png')
        plt.legend()
        plt.show()
        plt.clf()


