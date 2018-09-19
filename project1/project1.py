
'''
Written by Nathan West, Yonathan Mekonnen, and Derrick Adjei
09/09/18
CMSC 409
'''

import matplotlib.pyplot as pyplot
import numpy as npy
import sys

# generate male and female weights and heights (data)
male_weights = npy.random.normal(195.7, 20, 2000)
male_heights = npy.random.normal(5.9, .1, 2000)

female_weights = npy.random.normal(168.5, 20, 2000)
female_heights = npy.random.normal(5.4, .1, 2000)

f_female_above= int() #TP
f_female_below = int() #FP
s_female_above= int() #TP
s_female_below = int() #FP

f_male_above = int()  #TN
f_male_below = int()  #FN
s_male_above = int()  #TN
s_male_below = int()  #FN

def above_below(weight, height, key, g = "male"):
    above = int()
    below = int()
    if key == 'h':
        if height > 5.65:
            return True
        else:
            return False
    if key == 'w':
        if weight > (-142.65)*(height) + 990:
            return True
        else:
            return False

def acc_err(m_above, m_below, f_above, f_below):
    ACC = (m_above+f_below) / (m_above + m_below + f_above + f_below)
    ERR = 1 - ACC
    TP = m_above / (m_above+m_below)
    FP = f_above / (f_above+f_below)
    FN = m_below / (m_above+m_below)
    TN = f_below / (f_above+f_below)
    return ACC, ERR, TP, FP, FN, TN

# write data to file
file = open("data.txt", "w")
for i in range (0, 2000):
    file.write('%3.2f, %2.1f, 0\n' % (male_weights[i], male_heights[i]))
    if above_below(male_weights[i], male_heights[i], key = 'h') == True:
        abv = 1
        blo = 0
    else:
        abv = 0
        blo = 1
    f_male_above += abv
    f_male_below += blo
    if above_below(male_weights[i], male_heights[i], key = 'w') == True:
        abv = 1
        blo = 0
    else:
        abv = 0
        blo = 1
    s_male_above += abv
    s_male_below += blo
for i in range(0, 2000):
    file.write('%3.2f, %2.1f, 1\n' % (female_weights[i], female_heights[i]))
    if above_below(female_weights[i], female_heights[i], key = 'h') == True:
        abv = 1
        blo = 0
    else:
        abv = 0
        blo = 1
    f_female_above += abv
    f_female_below += blo
    if above_below(female_weights[i], female_heights[i], key = 'w', g= "female") == True:
        abv = 1
        blo = 0
    else:
        abv = 0
        blo = 1
    s_female_above += abv
    s_female_below += blo

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
yy = list()
xx = list()
for i in range(52,64):
    xx.append(i/10)
for i in xx:
    f = (-142.65)*(i) + 990
    yy.append(f)

pyplot.title("Both Weights and Heights Considered")
pyplot.xlabel("Male and Female Heights")
pyplot.ylabel("Male and Female Weights")
# calculate separation line
pyplot.scatter(male_heights, male_weights, color='b', marker='x')
pyplot.scatter(female_heights, female_weights, color ='r', marker='*')
pyplot.plot(xx, yy, color='g')
pyplot.show()


f_acc, f_err, f_TP, f_FP, f_FN, f_TN = acc_err(f_male_above, f_male_below, f_female_above, f_female_below)
# print(f_male_above, f_male_below, f_female_above, f_female_below)
print("----Scenario A----")
print("Acc: {0:.3f}".format(f_acc))
print("Err: {0:.3f}".format(f_err))
print("TP: {0:.3f}".format(f_TP))
print("FP: {0:.3f}".format(f_FP))
print("FN: {0:.3f}".format(f_FN))
print("TN: {0:.3f}".format(f_TN))

print("----Scenario B----")
s_acc, s_err, s_TP, s_FP, s_FN, s_TN = acc_err(s_male_above, s_male_below, s_female_above, s_female_below)
# print(s_male_above, s_male_below, s_female_above, s_female_below)
print("Acc: {0:.3f}".format(s_acc))
print("Err: {0:.3f}".format(s_err))
print("TP: {0:.3f}".format(s_TP))
print("FP: {0:.3f}".format(s_FP))
print("FN: {0:.3f}".format(s_FN))
print("TN: {0:.3f}".format(s_TN))
