'''
Written by Nathan West, Yonathan Mekonnen, Derrick Adjei
09/22/18
CMSC 409
'''

import random
import numpy as np
from numpy.random import seed
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

iter = 1000

def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * float(row[i])
    return 1 if activation >= 0 else 0

def train_weights(train, rate, epoch):
    weights = [random.randint(-0.5, 0.5), random.randint(-0.5, 0.5), random.randint(-0.5, 0.5)]

    true_neg, true_pos, false_neg, false_pos
    tot_error = 0
    count = 0

    while (tot_error > 10**-5 and count < iter):
        tot_error++
        for i in range(train):
            







male_heights = []
male_weights = []

female_heights = []
female_weights = []

males = []
females = []

file = open("data.txt", mode='r')

for i in range(2000):
    data = file.readline().split(",")
    tmp = []
    tmp.append(data[0])
    tmp.append(data[1])
    tmp.append(data[2])
    males.append(tmp)
    male_weights.append(data[0])
    male_heights.append(data[1])

# print(males)

for i in range(2000):
    data = file.readline().split(",")
    tmp = []
    tmp.append(data[0])
    tmp.append(data[1])
    females.append(tmp)
    female_weights.append(data[0])
    female_heights.append(data[1])



# testing shit
for row in males:
    prediction = predict(row, weights)
    print("Expected=%d, Predicted=%d" % (int(row[-1]), prediction))
