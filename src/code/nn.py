import numpy as np, pandas as pd
from matplotlib import pyplot as plt

class NeuralNetwork:
    
    def __init__(self, no_of_layers = 3, no_of_nodes = [1, 2, 1]):
        self.no_of_layers = no_of_layers
        self.no_of_nodes = no_of_nodes
        self.reg_lambda = 0.1
        self.learning_rate = 1.0
        self.epochs = 200
        self.initiate_network()
        self.print = False

    # Initializing all the weights 
    # np.random.randn() - normal distribution with mean = 0 and variance = 1
    def initiate_network(self):
        parameters = []
        for layer in range(self.no_of_layers-1):
            weight_matrix = np.random.randn(self.no_of_nodes[layer+1], self.no_of_nodes[layer]+1)
            parameters.append(weight_matrix)
        self.parameters = parameters

    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    
    # For loop to iterate over hidden layers.
    # np.matmul() - to compute all the forward operations.
    def feed_forward(self, x):
        activations = []
        a = np.append(1, x)
        for layer in range(self.no_of_layers-1):
            activations.append(a)
            z = np.matmul(self.parameters[layer], a)
            activation = self.sigmoid(z)
            a = np.append(1, activation)
        activations.append(activation)
        if self.print:
            print("--------------------Activations--------------------")
            for a in activations:
                print(a)
            print()
        return activation
    
    
    # np.delete() - to remove bias weights in cost calculation
    def compute_cost(self, data):
        cost = 0
        n = len(data)
        for x, y in data:
            f = self.feed_forward(x)
            j = -(y * np.log(f + np.finfo(float).eps) + (1-np.array(y)) * np.log(1-f+np.finfo(float).eps))
            cost += (np.sum(np.nan_to_num(j)) / n)
        reg_cost = 0.5*(1/n)*sum(np.sum(np.square(np.delete(w_layer, [0]*len(w_layer), axis=1))) for w_layer in self.parameters)
        total_cost = (cost + self.reg_lambda * reg_cost)
        if self.print:
            print("Average Instance cost J(I)      = ", cost)
            print("Total Regularized Cost(J)       = ", total_cost, "\n")
        return total_cost
    

    # **** This is implemented using arrays, vector and matrix operations.
    # **** For loops only used when iterating over hidden layers.
    def back_prop(self, x, y):
        z_layers = []
        activations = [x]
        a = np.append(1, x)

        for layer in range(len(self.parameters)):
            z = np.matmul(self.parameters[layer], a)
            z_layers.append(z)
            activations.append(self.sigmoid(z))
            a = np.append(1, self.sigmoid(z))
        
        deltas = {}
        delta = activations[-1] - y
        deltas[len(self.parameters)] = delta
        for layer in reversed(range(1, self.no_of_layers-1)):
            deltas[layer] = np.matmul(deltas[layer+1], np.delete(self.parameters[layer], [0]*len(self.parameters[layer]), axis = 1)) * activations[layer] * (1 - activations[layer])
        
        if self.print:
            print("-------------------Deltas of hidden and output layers--------------------")
            print(deltas, "\n")
            print("-------Gradients (not regularized) of network given a single input-------")
        gradients = []
        for layer in range(self.no_of_layers-1):
            gradient = np.array([np.append(1,activations[layer]),]*self.no_of_nodes[layer+1]) * np.array([deltas[layer+1],]*(1+self.no_of_nodes[layer])).transpose()
            gradients.append(gradient)
            if self.print:
                print(gradient,"\n")
        return gradients
    

    def gradientDescent(self, dataset):
        self.initiate_network()
        #print("Initial cost (J)           = ", self.compute_cost(dataset))
        #print()
        for epoch in range(self.epochs):
            self.update(dataset)
            #if epoch % 100 == 0:
            #    print("Epoch", epoch, ": Cost (J) = ", self.compute_cost(dataset))
        #print()
        #print("Final cost (J)             = ", self.compute_cost(dataset))
        #print("----------------------------------------------------------------------------\n")


    def update(self, dataset):
        gradients = []
        for x, y in dataset:
            gradients.append(self.back_prop(x, y))
        
        if self.print:
            print("-------------------Gradients (regularized)-------------------")
        grad = gradients[0]
        for j in range(len(grad)):
            for i in range(1, len(gradients)):
                grad[j] = grad[j] + gradients[i][j]
            grad[j] = grad[j] / len(dataset)
            grad[j][:,1:] = grad[j][:,1:] + ((self.reg_lambda * self.parameters[j][:,1:])/len(dataset))
            if self.print:
                print(grad[j],"\n")

        for layer in range(self.no_of_layers-1):
            self.parameters[layer] = self.parameters[layer] - (self.learning_rate * grad[layer])
    

    # Calculates the performance (accuracy, f1 score) of the model on a dataset
    def evaluate(self, dataset):
        predicted = []
        true = []
        for x, y in dataset:
            f = self.feed_forward(x)
            f_index = np.argmax(f)
            y_index = np.argmax(y)
            true.append(y_index)
            predicted.append(f_index)
        
        predicted_df = pd.DataFrame(predicted, columns=['majority'])
        true_df = pd.DataFrame(true, columns=['target'])
        x = pd.concat([predicted_df, true_df], axis = 1)
        total = len(x)
        n = x.target.nunique()        
        acc = f1 = p = r = 0
        
        for c in x.target.unique():
            tp = len(x[(x.target == x.majority) & (x.target == c)])
            tn = len(x[(x.target != c) & (x.majority != c)])
            fn = len(x[(x.target == c) & (x.majority != x.target)])
            fp = len(x[(x.majority == c) & (x.majority != x.target)])
            acc += ((tp+tn)/total)
            f1 += ((2*tp)/(2*tp+fn+fp))
            p += (tp/(tp+fp))
            r += (tp/(tp+fn))
        
        return acc/n, p/n, r/n, f1/n