import random
import math
import numpy as np


class Network:
    def __init__(self, inum, hnum, onum):
        # getting size of network from user
        self.inum = inum
        self.hnum = hnum
        self.onum = onum

        # initializing weight and bias matrix with random numbers
        self.first_layer_weight = np.random.uniform(0, 1, size = (inum, hnum))
        self.first_layer_bias = np.random.uniform(0, 1, size = (1, hnum))
        self.second_layer_weight = np.random.uniform(0, 1, size=(hnum, onum))
        self.second_layer_bias = np.random.uniform(0, 1, size=(1, onum))

        # initializing inputs, outputs and errors
        self.first_layer_net = np.array([0, 0])
        self.first_layer_output = np.array([0, 0])
        self.second_layer_net = 0
        self.second_layer_output = 0
        self.discretized_output = 0
        self.learning_rate = 0.03
        self.hidden_error = np.array([0, 0])
        self.output_error = 0
        self.delta_second_layer = 0
        self.delta_first_layer = np.array([0, 0])
        self.epoch = 80000

    # unipolar sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # unipolar sigmoid derivation function
    def sigdot(self, x):
        return 1/(1+np.exp(-x))

    # calculating outputs and updating weights
    def train(self, inp, outp):
        discretized_output = [0 for s in range(len(outp))]

        for i in range(self.epoch):
            self.first_layer_net = np.dot(inp, self.first_layer_weight)
            self.first_layer_net += self.first_layer_bias
            self.first_layer_output = self.sigmoid(self.first_layer_net)

            self.second_layer_net = np.dot(self.first_layer_output, self.second_layer_weight) + self.second_layer_bias
            self.second_layer_output = self.sigmoid(self.second_layer_net)

            self.output_error = outp - self.second_layer_output

            self.delta_second_layer = self.output_error * self.sigdot(self.second_layer_output)
            self.hidden_error = self.delta_second_layer.dot(self.second_layer_weight.T)

            self.delta_first_layer = self.hidden_error * self.sigdot(self.first_layer_output)

            self.second_layer_weight += self.learning_rate * (self.first_layer_output.T.dot(self.delta_second_layer))
            self.second_layer_bias += self.learning_rate * np.sum(self.delta_second_layer)
            self.first_layer_weight += self.learning_rate * (inp.T.dot(self.delta_first_layer))
            self.first_layer_bias += self.learning_rate * np.sum(self.delta_first_layer)

        # discretization
        for j in range(len(outp)):
            if self.second_layer_output[j] > 0.5:
                discretized_output[j] = 1
            else:
                discretized_output[j] = 0

        for z in range(len(outp)):
            print('Desired output:{0} | Discretized output:{2} | Predicted output:{1} '
                  .format(outp[z], self.second_layer_output[z], discretized_output[z]))


t = Network(2, 2, 1)
inp = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0.01, 0.01], [0.01, 0.99], [0.99, 0.01], [0.99, 0.99]])
out = np.array([[0], [1], [1], [0], [0], [1], [1], [0]])
t.train(inp, out)


