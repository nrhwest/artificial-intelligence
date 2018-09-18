
'''
Written by Nathan West, Yonathan Mekonnen, and Derrick Adjei
09/09/18
CMSC 409
'''

import matplotlib.pyplot as pyplot
import numpy as npy

# generate male and female weights and heights (data)
male_weights = npy.random.normal(195.7, 20, 2000)
male_heights = npy.random.normal(5.9, .1, 2000)

female_weights = npy.random.normal(168.5, 20, 2000)
female_heights = npy.random.normal(5.4, .1, 2000)

# write data to file
file = open("data.txt", "w")
for i in range (0, 2000):
    file.write('%3.2f, %2.1f, 0\n' % (male_weights[i], male_heights[i]))

for i in range(0, 2000):
    file.write('%3.2f, %2.1f, 1\n' % (female_weights[i], female_heights[i]))

file.close()

# plot male and female heights
x = [i for i in range(len(female_heights))]

pyplot.title("Male vs Female Heights")
pyplot.xlabel("Student IDs")
pyplot.ylabel("Male and Female Heights")
pyplot.scatter(x, female_heights, color='r')
pyplot.scatter(x, male_heights, color='b')
# calculate separation line
y = [2000 * 0 + 5.65 for i in x]
pyplot.plot(x, y, color='g')
pyplot.show()

# plot male and female weights and heights
import matplotlib.pyplot as pyplot2
yy = list()
xx = list()
for i in range(52,64):
    xx.append(i/10)
for i in range(12):
    print(i)
    f = (-28.65*(i+2)) + 375
    yy.append(f)
# yy.reverse()

pyplot2.title("Both Weights and Heights Considered")
pyplot2.xlabel("Male and Female Heights")
pyplot2.ylabel("Male and Female Weights")
# calculate separation line
pyplot2.scatter(male_heights, male_weights, color='b')
pyplot2.scatter(female_heights, female_weights, color ='r')
pyplot2.plot(xx, yy, color='g')
pyplot2.show()
