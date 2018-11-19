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
import pandas as pd

<<<<<<< HEAD
alpha = 0.3
numEpoch = 1000
=======
alpha = 0.30
numEpoch = 10
>>>>>>> 4e7d5b8189494d7ed6788380dff2a462c5861107

norm_data = list()
weights = []
train_size = 47

times = list()
output = list()

x2 = list()
y2 = list()

def graph():
    plt.scatter(x1, y1, c='r')
#     xx = np.array(range(-2, 12))
#     x = np.empty(15)
#     yy    = list()
#     for i in range(len(xx)):
#         x[i] = xx[i]/10
#     for each in x:
#         slope = -(weights[1]/weights[2])/(weights[0]/weights[1])
#         intercept = -weights[1]/weights[2]
#         yy.append((slope*each)+each)
#     plt.plot(x, yy, c='black')
    plt.show()

class Energy:
    def __init__(self, hour, consumption):
        self.hour = hour
        self.consumption = consumption

# def load(file):
#     # load data into energy objects
#     for i in range(16):
#         data    = file.readline().split(",")
#         energy  = Energy(float(data[0]), float(data[1]))
#         hr      = (energy.hour - 5.00)  / (20.00 - 5.00)
#         consum  = (energy.consumption - 2.0) / (10.0 - 2.0)
#         energy.hour = hr
#         energy.consumption = consum
#         times.append(hr)
#         output.append(consum)
#         norm_data.append(energy)

def load(file):
    # load data into energy objects
    test_objs = list()
    for i in range(16):
        data    = file.readline().split(",")
        energy  = Energy(float(data[0]), float(data[1]))
        hr      = (energy.hour - 5.00)  / (20.00 - 5.00)
        consum  = (energy.consumption - 2.0) / (10.0 - 2.0)
        energy.hour = hr
        energy.consumption = consum
        test_objs.append(energy)
        return test_objs

<<<<<<< HEAD
def predict(activation):
    if activation > 1:
        return 1
    else:
        return -1


# def calculate_accuracy(males, females, train_size):
#     tp = 0
#     fp = 0
#     tn = 0
#     fn = 0
#
#     for i in range(train_size, 2000):
#         net = (norm_data[i].weight * weights[1]) + weights[0]
#         if (net >= 1):
#             tp += 1
#         else:
#             fn += 1
#         net = (females[i].weight * weights[1]) + (females[i].height * weights[2]) + weights[0]
#         if sum < 0:
#             tn += 1
#         else:
#             fp += 1
#
#     tp = tp / (tp + fn)
#     fp = fp / (fp + tn)
#     tn = tn / (fp + tn)
#     fn = fn / (tp + fn)
#
#     accuracy = (tp + tn) / (tn + tp + fn + fp)
#     error = 1 - accuracy
#
#     print("TP = " + str(tp))
#     print("FP = " + str(fp))
#     print("TN = " + str(tn))
#     print("FN = " + str(fn))
#     print("Accuracy = " + str(accuracy))
#     print("Error = " + str(error) + "\n")
#

def fit_model(numEpoch, train_size, alpha):
    epoch = 0
    errorAmount = 1.0
    for i in range(3):
        weights.append(round(random.uniform(-0.5, 0.5), 2))
    while (errorAmount > 0.00001 and epoch < numEpoch):
        epoch += 1
        for i in range(train_size):

            net = (norm_data[i].consumption * weights[0]) + weights[1]
            desired = 1
            predictedOutput = predict(net)
            error = desired - predictedOutput

            # if net < 0:
            #     errorAmount += 1 / 4000
            # else:
            #     errorAmount += 0

            weights[0] += alpha * error * norm_data[i].consumption
            weights[1] += alpha * error
            print(weights)

            # if net >= 0:
            #     errorAmount += 1 / 4000
            # else:
            #     errorAmount += 0


train_data1 = open("data/train_data_1.txt", mode='r')
train_data2 = open("data/train_data_2.txt", mode='r')
train_data3 = open("data/train_data_3.txt", mode='r')
test_data = open("data/test_data_4.txt", mode='r')

load(train_data1)
# df = pd.read_csv(file1, delimiter=",")
# training_file1 = file1.read().split('\n')
# print(norm_data)

for i in range(3):
    if (i == 1):
        fit_model(numEpoch, 16, alpha)

=======
def predict(activation, expected):
    if activation >= expected:
        return activation
    else:
        return expected

def fit_model(numEpoch, train_size, alpha):
    epoch = 0               #number of training cycle
    errorAmount = 1.0
    for i in range(2):      #for input and bias
        weights.append(round(random.uniform(-0.5, 0.5), 2))
    while (errorAmount > 0.00001 and epoch < numEpoch):
        epoch += 1
        for i in range(train_size):
            bias    = 1 * weights[0]
            desired = output[i]
            net = (times[i] * weights[1]) + bias
            print("Time: {}".format(times[i]))
            predictedOutput = predict(net, desired)
            print("Net: {}, Expected: {}".format(net, desired))
            print("Prediction: {}".format(predictedOutput))
            error = desired - predictedOutput

            if net < 0:
                errorAmount += 1 / (16*3)
            else:
                errorAmount += 0

            weights[0] += alpha * error
            weights[1] += alpha * error * desired


def calculate_accuracy(males, females, train_size):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(train_size, 2000):
        sum = (males[i].weight * weights[1]) + (males[i].height * weights[2]) + weights[0]
        if (sum >= 0):
            tp += 1
        else:
            fn += 1
        sum = (females[i].weight * weights[1]) + (females[i].height * weights[2]) + weights[0]
        if sum < 0:
            tn += 1
        else:
            fp += 1

    tp = tp / (tp + fn)
    fp = fp / (fp + tn)
    tn = tn / (fp + tn)
    fn = fn / (tp + fn)

    accuracy = (tp + tn) / (tn + tp + fn + fp)
    error = 1 - accuracy

    print("TP = " + str(tp))
    print("FP = " + str(fp))
    print("TN = " + str(tn))
    print("FN = " + str(fn))
    print("Accuracy = " + str(accuracy))
    print("Error = " + str(error) + "\n")




file1 = open("data/train_data_1.txt", mode='r')
file2 = open("data/train_data_2.txt", mode='r')
file3 = open("data/train_data_3.txt", mode='r')
file4 = open("data/train_data_4.txt", mode='r')

one     = load(file1)
two     = load(file2)
three   = load(file3)
test    = load_test(file4)

fit_model(numEpoch, train_size, alpha)
# df = pd.read_csv(file1, delimiter=",")
# training_file1 = file1.read().split('\n')
# print(norm_data)
# for i in range(len(norm_data)):
#     print(i)
#     print(norm_data[i].hour)
#     print("{}\n".format(norm_data[i].consumption))
>>>>>>> 4e7d5b8189494d7ed6788380dff2a462c5861107

# graph()
