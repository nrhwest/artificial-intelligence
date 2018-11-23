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


def graph(obj_list):
    colors = ['b', 'g', 'r' , 'c', 'm','y' , 'k' , 'w']
    for each in range(len(obj_list)):
        if each == len(obj_list) - 1:
            x = list()
            y = list()
            for i in obj_list[each]:
                x.append(i.hour)
                y.append(i.consumption)
            plt.plot(x, y, c=colors[each])
            plt.xlabel('Hour')
            plt.ylabel('Consumption')
            plt.title('Prediction of Energy Consumption')

        else:
            for i in obj_list[each]:
                plt.scatter(i.hour, i.consumption, c=colors[each])
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
        consum = (energy.consumption - 2.0) / (10.0 - 2.0)
        energy.hour = hr
        energy.consumption = consum
        test_objs.append(energy)
    return test_objs

# def predict(activation, expected):
#     if activation >= expected:
#         return activation
#     else:
#         return -activation


def fit_model(instance, numEpoch, train_size, alpha, poly = 1):
    epoch = 0               # number of training cycle
    error_amount = 5
    # answers = list()

    # create randomized weights for inputs and bias
    for i in range(4):
        weights.append(round(random.uniform(-0.5, 0.5), 2))

    while (epoch < numEpoch and error_amount >= 5):
        epoch += 1

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

            # predictedOutput = predict(net, desired)
            #
            # try:
            #     error = desired - predictedOutput
            # except TypeError:
            #     error = d.Decimal(desired) - d.Decimal(predictedOutput)

            error = desired - net

            if (poly == 1):
                weights[0] += (alpha * error)
                weights[1] += (alpha * error) * instance[i].hour
            elif (poly == 2):
                weights[0] += (alpha * error)
                weights[1] += (alpha * error) * instance[i].hour
                weights[2] += (alpha * error) * instance[i].hour ** 2
            elif (poly == 3):
                weights[0] += (alpha * error)
                weights[1] += (alpha * error) * instance[i].hour
                weights[2] += (alpha * error) * instance[i].hour ** 2
                weights[3] += (alpha * error) * instance[i].hour ** 3

            # try:
            #     weights[0] += (alpha * error)
            #     weights[1] += (alpha * error) * instance[i].hour
            # except TypeError:
            #     weights[0] += (float(d.Decimal(alpha)) * float(d.Decimal(error)))
            #     weights[1] += (float(d.Decimal(alpha)) * float(d.Decimal(error))) * float(d.Decimal(instance[i].hour))

            # if epoch + 1 == numEpoch:
            #     answers.append(Energy(instance[i].hour, weights[1]))
    stuff = [test, answers]
    graph(stuff)
    return answers, errorAmount

# def model_predict(model, test, poly = 1):
#     answers = list()
#     bias    = 1 * test[0].consumption
#     for i in range(1,len(test)):
#         net = 0.0
#         desired = test[i].consumption
#         if poly == 1:
#             net = (test[i].hour * model[i].consumption) + bias
#         elif poly == 2:
#             try:
#                 net = (test[i].hour * model[i].consumption) + (test[i].hour * (model[i].consumption ** 2))+ bias
#             except OverflowError as o:
#                 net = d.Decimal(net)
#
#         elif poly == 3:
#             try:
#                 net = (test[i].hour * model[i].consumption) + (test[i].hour * (model[i].consumption ** 2))+ (test[i].hour * (model[i].consumption ** 3))+ bias
#             except OverflowError as o:
#                 net = d.Decimal(net)
#
#         predictedOutput = predict(net, desired)
#         answers.append(Energy(test[i].hour, weights[1]*float(predictedOutput)))



alpha = 0.3
numEpoch = 1000

weights = []
train_size = 16

file1 = open("data/train_data_1.txt", mode='r')
file2 = open("data/train_data_2.txt", mode='r')
file3 = open("data/train_data_3.txt", mode='r')
file4 = open("data/test_data_4.txt", mode='r')

test    = load(file4)
one     = load(file1)
model_one_1, error1 = fit_model(one, numEpoch, train_size, alpha, 1)
model_one_2, error2 = fit_model(one, numEpoch, train_size, alpha, 2)
model_one_3, error3 = fit_model(one, numEpoch, train_size, alpha, 3)
one_models = [model_one_1, model_one_2, model_one_3]
# for i in range(len(one_models)):
#     model_predict(one_models[i], test, i+1)

weights.clear()
two     = load(file2)
model_two_1, error1 = fit_model(two, numEpoch, train_size, alpha, 1)
model_two_2, error2 = fit_model(two, numEpoch, train_size, alpha, 2)
model_two_3, error3 = fit_model(two, numEpoch, train_size, alpha, 3)
two_models = [model_two_1, model_two_2, model_two_3]
# for i in range(len(two_models)):
#     model_predict(two_models[i], test, i+1)

weights.clear()
three     = load(file3)
model_three_1, error1 = fit_model(three, numEpoch, train_size, alpha, 1)
model_three_2, error2 = fit_model(three, numEpoch, train_size, alpha, 2)
model_three_3, error3 = fit_model(three, numEpoch, train_size, alpha, 3)
three_models = [model_three_1, model_three_2, model_three_3]
# for i in range(len(three_models)):
#     model_predict(three_models[i], test, i+1)
