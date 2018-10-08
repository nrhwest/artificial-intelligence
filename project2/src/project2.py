'''
Written by Nathan West, Yonathan Mekonnen, Derrick Adjei
09 / 22 / 18
CMSC 409
'''

import random
import numpy as np
import sys
import matplotlib.pyplot as plt

class Person:
    def __init__(self, weight, height, gender):
        self.weight = weight
        self.height = height
        self.gender = gender

alpha = 0.30
n_epoch = 1000
gender = True

males = []
females = []
weights = []

file = open("data.txt", mode = 'r')

# read in male data
for i in range(2000):
    data = file.readline().split(",")
    person = Person(float(data[0]), float(data[1]), data[2])
    w = 1 / (250.00 - 120.00) * (person.weight - 120.00)
    h = 1 / (7.0 - 5.0) * (person.height - 5.0)# plt.scatter(w, h, c = 'r')
    males.append(person)

# read in female data
for i in range(2000):
    data = file.readline().split(",")
    person = Person(float(data[0]), float(data[1]), data[2])
    w = 1 / (250.00 - 120.00) * (person.weight - 120.00)
    h = 1 / (7.0 - 5.0) * (person.height - 5.0)# plt.scatter(w, h, c = 'b')
    females.append(person)

# plt.show()
file.close()

for i in range(3):
    weights.append(round(random.uniform(-0.5, 0.5), 2))


def predict(activation):
    if activation > 0:
        return 1
    else :
        return 0

file2 = open("output.txt", mode = 'w')


# train for 25%
epoch = 0
sum_error = 1.0
while (sum_error > 0.00001 and epoch < n_epoch):
    epoch += 1
    for i in range(500):

        net = (males[i].weight * weights[0]) + (males[i].height * weights[1]) + weights[2]
        desired = 1
        prediction = predict(net)
        # print("Expected=%d, Predicted=%d" % (desired, prediction))
        error = desired - prediction
        # print("from male prediction = " + str(prediction) + "\n")

        sum_error += 1 / 4000 if net < 0 else 0

        weights[0] += alpha * error
        weights[1] += alpha * error * males[i].weight
        weights[2] += alpha * error * males[i].height
        print(str(weights) + "\n")

        # print(weights, sep = ' ', end = '\n', file = sys.stdout, flush = False)
        # file2.write("in male = " + "weights = " + str(weights) + "  error = " + str(error) + "\n")

        # print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, alpha, error))

        net = (females[i].weight * weights[1]) + (females[i].height * weights[2]) + weights[0]

        desired = 0
        prediction = predict(net)
        error = desired - prediction
        # print("from female prediction = " + str(prediction) + "\n")

        weights[0] += alpha * error
        weights[1] += alpha * error * females[i].weight
        weights[2] += alpha * error * females[i].height
        print(str(weights) + "\n")

        sum_error += 1 / 4000 if net < 0 else 0

        # file2.write("in female = " + "weights = " + str(weights) + "  error = " + str(error) + "\n")

# weights = train_weights(males, females, 100, alpha, n_epoch, gender)
print(weights)

while (sum_error > 0.00001 and epoch < n_epoch):
    epoch += 1
    for i in range(500):

        net = (males[i].weight * weights[0]) + (males[i].height * weights[1]) + weights[2]
        desired = 1
        prediction = predict(net)
        # print("Expected=%d, Predicted=%d" % (desired, prediction))
        error = desired - prediction
        # print("from male prediction = " + str(prediction) + "\n")

        sum_error += 1 / 4000 if net < 0 else 0

        weights[0] += alpha * error
        weights[1] += alpha * error * males[i].weight
        weights[2] += alpha * error * males[i].height
        print(str(weights) + "\n")

        # print(weights, sep = ' ', end = '\n', file = sys.stdout, flush = False)
        # file2.write("in male = " + "weights = " + str(weights) + "  error = " + str(error) + "\n")

        # print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, alpha, error))

        net = (females[i].weight * weights[1]) + (females[i].height * weights[2]) + weights[0]

        desired = 0
        prediction = predict(net)
        error = desired - prediction
        # print("from female prediction = " + str(prediction) + "\n")

        weights[0] += alpha * error
        weights[1] += alpha * error * females[i].weight
        weights[2] += alpha * error * females[i].height
        print(str(weights) + "\n")

        sum_error += 1 / 4000 if net < 0 else 0

        # file2.write("in female = " + "weights = " + str(weights) + "  error = " + str(error) + "\n")
