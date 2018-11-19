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
alpha = 0.30
numEpoch = 10

norm_data = list()
weights = []
train_size = 16

times = list()
output = list()

x2 = list()
y2 = list()


=======
>>>>>>> 21bf778f514ff4ea82a132ea3a2817d8f3fe14ff
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

def load(file):
    # load data into energy objects
    test_objs = list()
    for i in range(16):
        data    = file.readline().split(",")
        energy  = Energy(float(data[0]), float(data[1]))
        hr      = (energy.hour - 5.00)  / (20.00 - 5.00)
        consum  = (energy.consumption - 2.0) / (10.0 - 2.0)
        energy.hour = hr
        times.append(hr)
        energy.consumption = consum
        test_objs.append(energy)
    return test_objs

def predict(activation, expected):
    if activation >= expected:
        return activation
    else:
        return expected

def fit_model(instance, numEpoch, train_size, alpha):
    weights = list()
    epoch = 0               #number of training cycle
    errorAmount = 1.0
    for i in range(2):      #for input and bias
        weights.append(round(random.uniform(-0.5, 0.5), 2))
    while (errorAmount > 0.00001 and epoch < numEpoch):
        epoch += 1

        print("Epoch : ", epoch)
        for i in range(train_size):
<<<<<<< HEAD
            bias = 1 * weights[0]
            desired = test[i].consumption
            net = (times[i] * weights[1]) + bias
            print("Time: {}".format(times[i]))
=======
            bias    = 1 * weights[0]
            desired = instance[i].consumption
            net = (instance[i].hour * weights[1]) + bias
            print("Time: {}".format(instance[i].hour))
>>>>>>> 21bf778f514ff4ea82a132ea3a2817d8f3fe14ff
            predictedOutput = predict(net, desired)
            print("Net: {}, Expected: {}".format(net, desired))
            print("Prediction: {}".format(predictedOutput))
            error = desired - predictedOutput

            if net > 0:
                errorAmount += 1 / (16*3)
            else:
                errorAmount += 0

            weights[0] += alpha * error
<<<<<<< HEAD
            weights[1] += alpha * error * desired
            print("weights = ", weights)
            print()


# def calculate_accuracy(males, females, train_size):
#     tp = 0
#     fp = 0
#     tn = 0
#     fn = 0
#
#     for i in range(train_size, 2000):
#         sum = (males[i].weight * weights[1]) + (males[i].height * weights[2]) + weights[0]
#         if (sum >= 0):
#             tp += 1
#         else:
#             fn += 1
#         sum = (females[i].weight * weights[1]) + (females[i].height * weights[2]) + weights[0]
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
=======
            weights[1] += alpha * error * instance[i].hour
>>>>>>> 21bf778f514ff4ea82a132ea3a2817d8f3fe14ff

alpha = 0.30
numEpoch = 1000

weights = []
train_size = 47

file1 = open("data/train_data_1.txt", mode='r')
file2 = open("data/train_data_2.txt", mode='r')
file3 = open("data/train_data_3.txt", mode='r')
file4 = open("data/train_data_4.txt", mode='r')

one     = load(file1)
fit_model(one, numEpoch, train_size, alpha)

two     = load(file2)

three   = load(file3)

test    = load(file4)

# df = pd.read_csv(file1, delimiter=",")
# training_file1 = file1.read().split('\n')
# print(norm_data)
# for i in range(len(norm_data)):
#     print(i)
#     print(norm_data[i].hour)
#     print("{}\n".format(norm_data[i].consumption))

# graph()
