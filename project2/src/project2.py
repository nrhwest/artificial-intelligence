'''
Written by Nathan West, Yonathan Mekonnen, Derrick Adjei
09/22/18
CMSC 409
'''

import matplotlib.pyplot as pyplot
import random
import numpy as npy
import sys

def perceptron():
    pass


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


    # ltotal = len(open('data.txt', 'r').read().split('\n'))
    # print("ltotal = ", ltotal)
    # lim_75 = int(ltotal * .75)
    # print("lim_75 = ", lim_75)
    #
    # for line
    # if lim_75 < 1:
    #     lim_75 = 1
    #
    #
    #
    # for line in file:
    #     print(line)




if __name__ == "__main__": main()
