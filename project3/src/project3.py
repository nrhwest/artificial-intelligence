'''
Written by Nathan West, Yonathan Mekonnen, Derrick Adjei
11/07/18
CMSC 409

The dataset comes from the randomly generated data used in the first Project

Nathan wrote the code for setting up our data structure (class object)
Nathan, Yonathan wrote the code for the perceptron
Derrick, Yonathan wrote the code for plotting the separation lines
'''

'''
Questions
Are we predicting the consumption for day 4 between the hours of 5am-8pm?
What are we using the 3 different architectures for? training, then predicting day 4?
'''

import random
import numpy as np
import matplotlib.pyplot as plt
import decimal as d
import math


def graph(obj_list, poly, title):
    colors = ['b', 'g', 'r' , 'c', 'm','y' , 'k' , 'w']
    plt.xlabel('Hour')
    plt.ylabel('Consumption')
    plt.title('Prediction of Energy w/ degree '+str(poly) + ' ' + title)
    x = list()
    for each in obj_list[0]:
        x.append(each.hour)
    x = np.array(x)
    for each in obj_list[0]:
        plt.scatter(each.hour, each.consumption, c='b')
        if poly == 3:
            y = (weights[1] * x) + (weights[2] * (x ** 2)) + (weights[3] * (x ** 3)) + weights[0]
        elif poly == 2:
            y = (weights[1] * x) + (weights[2] * (x ** 2))  + weights[0]
        elif poly == 1:
            y = (weights[1] * x) + weights[0]
        plt.plot(x, y, c='r')
    plt.show()

class Energy:
    def __init__(self, hour, consumption):
        self.hour = hour
        self.consumption = consumption

def load(file):
    # load data into energy objects
    test_objs = list()
    for i in range(16):
        data = file.readline().split(",")
        energy = Energy(float(data[0]), float(data[1]))
        hr = (energy.hour - 5.00) / (20.00 - 5.00)
        energy.hour = hr
        test_objs.append(energy)
    return test_objs



def fit_model(instance, numEpoch, train_size, alpha, poly = 1, indc = ''):
    epoch = 0               # number of training cycle
    error_amount = 5
    # create randomized weights for inputs and bias
    for i in range(4):
        weights.append(round(random.uniform(-0.5, 0.5), 2))

    while (epoch < numEpoch and error_amount >= 5):
        epoch += 1
        total_error = 0
        for i in range(train_size):
            bias = 1 * weights[0]
            desired = instance[i].consumption
            net = 0.0

            if poly == 1:
                net = (instance[i].hour * weights[1]) + bias
            elif poly == 2:
                net = (instance[i].hour * weights[1]) + ((instance[i].hour ** 2) * weights[2]) + bias
            elif poly == 3:
                net = (instance[i].hour * weights[1]) + ((instance[i].hour ** 2) * weights[2]) + ((instance[i].hour ** 3) * weights[3]) + bias

            error = desired - net

            total_error += error ** 2

            if (poly == 1):
                weights[0] += (alpha * error)
                weights[1] += (alpha * error) * instance[i].hour
            elif (poly == 2):
                weights[0] += (alpha * error)
                weights[1] += (alpha * error) * instance[i].hour
                weights[2] += (alpha * error) * (instance[i].hour ** 2)
            elif (poly == 3):
                weights[0] += (alpha * error)
                weights[1] += (alpha * error) * instance[i].hour
                weights[2] += (alpha * error) * (instance[i].hour ** 2)
                weights[3] += (alpha * error) * (instance[i].hour ** 3)

    print('Degree of Function: ' + str(poly))
    print('Error:', total_error)
    print('Run weights: ', weights)
    print('\n')
    graph([instance], poly, indc)




alpha = 0.1
numEpoch = 10000

weights = []
train_size = 16

file1 = open("data/train_data_1.txt", mode='r')
file2 = open("data/train_data_2.txt", mode='r')
file3 = open("data/train_data_3.txt", mode='r')
file4 = open("data/test_data_4.txt", mode='r')

one     = load(file1)
two     = load(file2)
three   = load(file3)
test    = load(file4)

train_data = [one, two, three]
for i in range(1,4):
    weights.clear()
    for x in range(0,3):
        fit_model(train_data[x], numEpoch, train_size, alpha, i)
    fit_model(test, numEpoch, train_size, alpha, i, 'TEST DAY 4')
