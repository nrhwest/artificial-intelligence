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

alpha = 0.30
numEpoch = 1000

norm_data = list()
weights = []
train_size = 500
x1 = list()
y1 = list()

x2 = list()
y2 = list()


class Energy:
    def __init__(self, hour, consumption):
        self.hour = hour
        self.consumption = consumption
        # self.day = day


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


def load(file):
    # load data into energy objects
    for i in range(16):
        data = file.readline().split(",")
        energy = Energy(float(data[0]), float(data[1]))
        hr = 1 / (24.00 - 1.00) * (energy.hour - 1.00)
        consum = 1 / (10.0 - 1.0) * (energy.consumption - 1.0)
        energy.hour = hr
        energy.consumption = consum
        x1.append(hr)
        y1.append(consum)
        # plt.scatter(hr, consum, c='r')
        norm_data.append(energy)


def predict(activation):
    if activation > 0:
        return 1
    else:
        return 0


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


# def fit_model(numEpoch, train_size, alpha):
#     # train with 25% data for hard activation
#     epoch = 0
#     errorAmount = 1.0
#     for i in range(3):
#         weights.append(round(random.uniform(-0.5, 0.5), 2))
#         while (errorAmount > 0.00001 and epoch < numEpoch):
#             epoch += 1
#             for i in range(train_size):
#
#                 net = (males[i].weight * weights[1]) + (males[i].height * weights[2]) + weights[0]
#                 desired = 1
#                 predictedOutput = predict(net)
#                 error = desired - predictedOutput
#
#                 if net < 0:
#                     errorAmount += 1 / 4000
#                 else:
#                     errorAmount += 0
#
#                 weights[0] += alpha * error
#                 weights[1] += alpha * error * males[i].weight
#                 weights[2] += alpha * error * males[i].height
#
#                 if net >= 0:
#                     errorAmount += 1 / 4000
#                 else:
#                     errorAmount += 0


file1 = open("data/train_data_1.txt", mode='r')
load(file1)
# df = pd.read_csv(file1, delimiter=",")
# training_file1 = file1.read().split('\n')
# print(norm_data)
for i in range(len(norm_data)):
    print(i)
    print(norm_data[i].hour)
    print("{}\n".format(norm_data[i].consumption))

graph()
