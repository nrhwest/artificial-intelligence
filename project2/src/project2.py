'''
Written by Nathan West, Yonathan Mekonnen, Derrick Adjei
09 / 22 / 18
CMSC 409

The dataset comes from the randomly generated data used in the first Project

Nathan wrote the code for setting up our data structure (class object)
Nathan, Yonathan wrote the code for the perceptron
Derrick, Yonathan wrote the code for plotting the separation lines
'''

import random
import numpy as np
import matplotlib.pyplot as plt


class Person:
    def __init__(self, weight, height, gender):
        self.weight = weight
        self.height = height
        self.gender = gender


alpha = 0.30
numEpoch = 1000
gender = True

males = []
females = []
weights = []

file = open("data.txt", mode='r')

x1 = list()
y1 = list()

x2 = list()
y2 = list()

<<<<<<< HEAD
def graph():
    plt.scatter(x1, y1, c = 'r')
    plt.scatter(x2, y2, c = 'b')
=======

def graph():
    plt.scatter(x1, y1, c='r')
    plt.scatter(x2, y2, c='b')
>>>>>>> b3b1a49a7c0b1bf1b848e70dee051159fb852a1c
    xx = np.array(range(-2, 12))
    x = np.empty(15)
    yy = list()
    for i in range(len(xx)):
        x[i] = xx[i]/10
<<<<<<< HEAD
    count = 0
    for each in x:
        slope = -(weights[0]/weights[2])/(weights[0]/weights[1])
        intercept = -weights[0]/weights[2]
        yy.append((slope*each)+each)
    plt.plot(x, yy, c = 'black')
    plt.show()

=======
    for each in x:
        slope = -(weights[1]/weights[2])/(weights[0]/weights[1])
        intercept = -weights[1]/weights[2]
        yy.append((slope*each)+each)
    plt.plot(x, yy, c='black')
    plt.show()


>>>>>>> b3b1a49a7c0b1bf1b848e70dee051159fb852a1c
def load():
    # read in male data
    for i in range(2000):
        data = file.readline().split(",")
        person = Person(float(data[0]), float(data[1]), data[2])
        w = 1 / (250.00 - 120.00) * (person.weight - 120.00)
        h = 1 / (7.0 - 5.0) * (person.height - 5.0)
        x1.append(w)
        y1.append(h)
<<<<<<< HEAD
        plt.scatter(w, h, c = 'r')
=======
        plt.scatter(w, h, c='r')
>>>>>>> b3b1a49a7c0b1bf1b848e70dee051159fb852a1c
        males.append(person)

    # read in female data
    for i in range(2000):
        data = file.readline().split(",")
        person = Person(float(data[0]), float(data[1]), data[2])
        w = 1 / (250.00 - 120.00) * (person.weight - 120.00)
        h = 1 / (7.0 - 5.0) * (person.height - 5.0)
        x2.append(w)
        y2.append(h)
<<<<<<< HEAD
        plt.scatter(w, h, c = 'b')
        females.append(person)

# plt.show()
# file.close()
=======
        plt.scatter(w, h, c='b')
        females.append(person)


load()
>>>>>>> b3b1a49a7c0b1bf1b848e70dee051159fb852a1c

load()

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


# train with 25% data for hard activation
epoch = 0
errorAmount = 1.0
train_size = 500
for i in range(3):
    weights.append(round(random.uniform(-0.5, 0.5), 2))
while (errorAmount > 0.00001 and epoch < numEpoch):
    epoch += 1
    for i in range(train_size):

        net = (males[i].weight * weights[1]) + (males[i].height * weights[2]) + weights[0]
        desired = 1
        predictedOutput = predict(net)
        error = desired - predictedOutput

        if net < 0:
            errorAmount += 1 / 4000
        else:
            errorAmount += 0

        weights[0] += alpha * error
        weights[1] += alpha * error * males[i].weight
        weights[2] += alpha * error * males[i].height

        net = (females[i].weight * weights[1]) + (females[i].height * weights[2]) + weights[0]

        desired = 0
        predictedOutput = predict(net)
        error = desired - predictedOutput

        weights[0] += alpha * error
        weights[1] += alpha * error * females[i].weight
        weights[2] += alpha * error * females[i].height

        if net >= 0:
            errorAmount += 1 / 4000
        else:
            errorAmount += 0

graph()
print("Accuracy for 25% hard activation")
calculate_accuracy(males, females, train_size+1)


# train with 75% for hard activation
epoch = 0
errorAmount = 1.0
train_size = 1500
for i in range(3):
    weights.append(round(random.uniform(-0.5, 0.5), 2))
while (errorAmount > 0.00001 and epoch < numEpoch):
    epoch += 1
    for i in range(train_size):

        net = (males[i].weight * weights[1]) + (males[i].height * weights[2]) + weights[0]
        desired = 1
        predictedOutput = predict(net)
        error = desired - predictedOutput

        if net < 0:
            errorAmount += 1 / 4000
        else:
            errorAmount += 0

        weights[0] += alpha * error
        weights[1] += alpha * error * males[i].weight
        weights[2] += alpha * error * males[i].height

        net = (females[i].weight * weights[1]) + (females[i].height * weights[2]) + weights[0]

        desired = 0
        predictedOutput = predict(net)
        error = desired - predictedOutput

        weights[0] += alpha * error
        weights[1] += alpha * error * females[i].weight
        weights[2] += alpha * error * females[i].height

<<<<<<< HEAD
<<<<<<< HEAD
        errorAmount += 1 / 4000 if net < 0 else 0
graph()
=======
=======
        errorAmount += 1 / 4000 if net < 0 else 0
>>>>>>> b3b1a49a7c0b1bf1b848e70dee051159fb852a1c
        if net >= 0:
            errorAmount += 1 / 4000
        else:
            errorAmount += 0

<<<<<<< HEAD
>>>>>>> 8050bfdbb174da65d509852e6ba1727f1b219fa1
=======
graph()
>>>>>>> b3b1a49a7c0b1bf1b848e70dee051159fb852a1c
print("Accuracy for 75% hard activation")
calculate_accuracy(males, females, train_size+1)

# train with 25% for soft activation
epoch = 0
errorAmount = 1.0
train_size = 500
for i in range(3):
    weights.append(round(random.uniform(-0.5, 0.5), 2))
while (errorAmount > 0.00001 and epoch < numEpoch):
    epoch += 1
    for i in range(train_size):

        net = (males[i].weight * weights[1]) + (males[i].height * weights[2]) + weights[0]
        desired = 1

        predictedOutput = float(1 / (1 + np.exp(-net)))
        error = desired - predictedOutput

        weights[0] += alpha * error
        weights[1] += alpha * error * males[i].weight
        weights[2] += alpha * error * males[i].height

        if net < 0:
            errorAmount += 1 / 4000
        else:
            errorAmount += 0

        net = (females[i].weight * weights[1]) + (females[i].height * weights[2]) + weights[0]
        desired = 0
        predictedOutput = float(1 / (1 + np.exp(-net)))

        error = desired - predictedOutput

        weights[0] += alpha * error
        weights[1] += alpha * error * females[i].weight
        weights[2] += alpha * error * females[i].height

        if net >= 0:
            errorAmount += 1 / 4000
        else:
            errorAmount += 0

graph()
print("Accuracy for 25%  soft activation")
calculate_accuracy(males, females, train_size+1)

# train with 75% for soft activation
epoch = 0
errorAmount = 1.0
train_size = 1500
for i in range(3):
    weights.append(round(random.uniform(-0.5, 0.5), 2))
while (errorAmount > 0.00001 and epoch < numEpoch):
    epoch += 1
    for i in range(train_size):

        net = (males[i].weight * weights[1]) + (males[i].height * weights[2]) + weights[0]
        desired = 1
        predictedOutput = float(1 / (1 + np.exp(-net)))
        error = desired - predictedOutput

        weights[0] += alpha * error
        weights[1] += alpha * error * males[i].weight
        weights[2] += alpha * error * males[i].height

        if net < 0:
            errorAmount += 1 / 4000
        else:
            errorAmount += 0

        net = (females[i].weight * weights[1]) + (females[i].height * weights[2]) + weights[0]

        desired = 0
        predictedOutput = float(1 / (1 + np.exp(-net)))
        error = desired - predictedOutput

        weights[0] += alpha * error
        weights[1] += alpha * error * females[i].weight
        weights[2] += alpha * error * females[i].height
        # print(str(weights) + "\n")

        if net >= 0:
            errorAmount += 1 / 4000
        else:
            errorAmount += 0
graph()
print("Accuracy for 75%  soft activation")
calculate_accuracy(males, females, train_size+1)
