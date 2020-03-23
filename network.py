import numpy as np
import time
from utils import *
import pickle

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
        
        print("\n ________________________________________ ")
        print("\n Predictions : ")
        for i, elt in enumerate(result):
            if elt[0][0] >= 0.5:
                elt[0][0] = 1
            else:
                elt[0][0] = 0
            print(" ",i+1,"=>",elt)
        print(f" min error =  {min(self.errors):.7f}\n")
        print(" ________________________________________ END")
    
    # Organize data
    def struct_train(self, xtrain, ytrain):
        xtrain = [np.array([elt]) for elt in xtrain]
        ytrain = [np.array([elt]) for elt in ytrain]
        return xtrain, ytrain


    # train the network
    def train(self, x_train, y_train, 
        epochs=0, learning_rate=0, 
        graphic=False, name='gradient_descent'):

        print("\n TRAINING REPORT \n")
        # Initial call to print 0% progress for the progress bar
        ProgressBar(0, epochs, prefix = 'Progress:', suffix = 'Ok', length = 30)
        
        # sample dimension first
        samples = len(x_train)
        x_train, y_train = self.struct_train(x_train, y_train)

        # Set time to calculate script duration
        begin = time.time()

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
            #print('epoch(s) %d/%d => err = %.10f' % (i+1, epochs, err))
            #print(f'epoch {i+1}/epochs => error={err}')
            
            # setting values of training report
            self.epoch.append(i)
            self.errors.append(err)
        
            # Update Progress Bar
            ProgressBar(i + 1, epochs, prefix = 'Progress:', suffix = err, length = 30)
        
        end = time.time()
        duration = end - begin
        print(f"\n Duration : {duration:.3f} sec")

        if graphic:
            self.get_train_graph(name)
            
            
            
    # sending training report
    def final_state(self):
        return (self.errors, self.epoch)

    
    def get_train_graph(self, name='gradient_descent'):
        errors, iterations = self.final_state()
        import matplotlib.pyplot as plt
        #plt.grid(True)
        plt.title("Min = "+str(min(errors)))
        plt.plot(iterations, errors, label="Gradient")
        plt.xlabel('iterations')
        plt.ylabel('error')
        plt.savefig(name+'.png')
        plt.legend()
        plt.show()

    def save_model(self, model, name="default"):
        pass
    
    def upload_model(self, model):
        pass