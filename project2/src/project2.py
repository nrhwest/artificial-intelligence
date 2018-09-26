'''
Written by Nathan West, Yonathan Mekonnen, Derrick Adjei
09/22/18
CMSC 409
'''

import random
import numpy as np
from numpy.random import seed
import pandas as import pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import rcParams
rcParams["figure.figsize"] = 10,5
%matplotlib inline

class Perceptron(object):

    '''
    Parameters:
        eta: Learning rate (between 0.01 and 1.0)
        n_iter: Passes over the training set

    Attributes:
        w: Weights after fitting
        error_lists: number of misclassifications in every epoch
    '''

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        '''
        X: training features
        y: desired output
        '''

        self.w_ = np.zeros(1 + X.shape[i])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict_hard_activation(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0)
            self.errors_.append(errors)
            
        return self

        def net_input(self, X):
            '''
            calculate net net_input
            '''
            return np.dot(X, self.w_[1:]) + self.w_[0]

        def predict_hard_activation(self, X):
            '''
            return class label (hard activition)
            ''''
            return np.where(self.net_input(X) >= 0.0, 1, 0)

        def predict_hard_activation(self, X):
            '''
            return class label (soft activition)
            ''''
            return np.where(self.net_input(X) >= 0.0, 1, -1)

def main():
    train_data = []
    test_data = []

    file = open('data.txt', 'r')
    for line in file:
        r = random.random()
        if r < 0.75:
            train_data.append(line)
        else:
            test_data.append(line)

    file.close()

    x = []
    y = []
    desired_output =[]

    for line in train_data:
        parsed_data = line.split(',')
        data = " ".join(parsed_data[0], parsed_data[1])
        # x.append(join(parsed_data[0], parsed_data[1])
        y.append(parsed_data[1])
        desired_output.append(parsed_data[2])



    print(x)
    print(y)
    print(desired_output)

if __name__ == "__main__": main()
