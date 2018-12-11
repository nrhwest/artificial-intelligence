'''
Written by Nathan West
12/09/18
CMSC 409 - Final Exam

2.2 Naive Bayes Classifier
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics as stat


def graph(class_summaries):
    plt.xlabel('Mean')
    plt.ylabel('Standard Deviation')
    bins = 20
    for key, value in class_summaries.items():
        mean = value[0]
        stdev = value[1]

        x = np.random.normal(mean, stdev, 1000)
        count, bins, ignored = plt.hist(x, 20, normed=True)
        plt.plot(bins, normal_distribution(mean, stdev, bins), linewidth=3, color='r')
        plt.show()


def normal_distribution(mean, stdev, number):
        return 1 / (stdev * np.sqrt(2 * np.pi)) * np.exp( - (number - mean)**2 / (2 * stdev**2) )


def calc_probabilites(class_res, number):
    probabilites = {}

    for key, value in class_res.items():
        probabilites[key] = 1
        mean = value[0]
        stdev = value[1]

        for i in range(len(value)):
            probabilites[key] = normal_distribution(mean, stdev, number)

    return probabilites


def data_preprocessing(data_frame):
    dataset = {}
    for row in data_frame.values:
        record = row.tolist()
        class_num = record[-1]
        if (class_num not in dataset):
            dataset[class_num] = []
        dataset[class_num].append(record[0])

    return dataset


def calc_accuracy(predictions, targets):
    sum = 0
    for i in range(len(predictions)):
        if (predictions[i] == targets[i]):
            sum += 1

    return (sum/float(len(targets))) * 100.0


def main():
    df = pd.read_csv("Ex2_train.txt", sep=',')

    training_set = data_preprocessing(df)

    class_summaries = {}
    for key in training_set:
        class_summaries[key] = (stat.mean(training_set[key]), stat.stdev(training_set[key]))

    test_set = pd.read_csv("Ex2_test.txt", sep=',')

    graph_data = {k: [] for k in range(1, 4)}

    predictions = []
    for row in test_set.values:
        vector = row.tolist()
        probabilites = calc_probabilites(class_summaries, vector[0])
        print(probabilites)

        for key, val in graph_data.items():
            graph_data[key].append(probabilites[key])

        pred = max(probabilites, key=probabilites.get)
        predictions.append(int(pred))

    target = np.array(test_set[' class']).tolist()

    graph(class_summaries)
    acc = calc_accuracy(predictions, target)

    print("Accuracy of Naive Bayes Classifier : {}%".format(int(acc)))


if __name__ == '__main__':
    main()
