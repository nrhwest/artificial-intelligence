
'''
Written by Nathan West, Yonathan Mekonnen, and Derrick Adjei
09/09/18
CMSC 409
'''
import matplotlib.pyplot as pyplot
import numpy as npy

# generate male and female weights and heights (data)
male_weights = npy.random.normal(195.7, 20, 2000)
male_heights = npy.random.normal(5.9, .2, 2000)

female_weights = npy.random.normal(168.5, 20, 2000)
female_heights = npy.random.normal(5.4, .2, 2000)

# write data to file
file = open("data.txt", "w")
for i in range (0, 2000):
    file.write('%3.2f, %2.1f, 0\n' % (male_weights[i], male_heights[i]))

for i in range(0, 2000):
    file.write('%3.2f, %2.1f, 1\n' % (female_weights[i], female_heights[i]))

file.close()

# plot male and female heights
x = [i for i in range(len(female_heights))]

pyplot.scatter(x, female_heights, color='r')
pyplot.scatter(x, male_heights, color='b')
y = [2000 * 0 + 5.65 for i in x]
pyplot.plot(x, y, color='g')
pyplot.show()
