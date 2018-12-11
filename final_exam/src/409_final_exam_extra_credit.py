'''
Written by Nathan West
CMSC 409 - Final Exam Extra Credit
12/11/18

This final exam extra credit implements the Winner-Take-All algorithm.
'''

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def graph(weights, data, results):
    x_weights = []
    y_weights = []
    norm_weights = []

    for i in weights:
        normalized = normalize(i)
        norm_weights.append(normalized)

    for i in range(0, len(norm_weights)):
        x_weights.append(norm_weights[i][0])
        y_weights.append(norm_weights[i][1])

    plt.scatter(data['x1'], data['x2'], label='data', c=results, cmap=cm.brg)
    plt.plot(x_weights, y_weights, 'x', c='b', label='clusters', markersize=10)
    plt.ylabel('X2')
    plt.xlabel('X1')
    plt.legend()
    plt.title('Kohonen Network Clustering')
    plt.show()


def euclidean_distance(clusters, distances, weights, vector):
    for i in range(clusters):
        distances[i] = 0
        for j in range(len(vector)):
            val = (weights[i][j] - vector[j]) ** 2
            distances[i] += val


def wta_clustering(tdm, training_size, alpha, clusters):
    weights = []
    neuron2_static_weights = [[4.5, 3.5], [-6.5, -2.8]]
    neuron3_static_weights = [[9.5, 2.8], [-11.5, -4.3], [5.87, 2.71]]
    neuron7_static_weights = [[4.5, 2.1], [4.04, 2.88], [-9.80, -3.50],
            [4.9, 1.27],
            [-9.67, -3.84],
            [4.5, 2.1],
            [4.4, 2.38]]

    if clusters == 2:
        weights = neuron2_static_weights
    elif clusters == 3:
        weights = neuron3_static_weights
    elif clusters == 7:
        weights = neuron7_static_weights
    else:
        weights = np.random.rand(clusters, len(tdm[0]))

    final = list()
    distances = np.zeros(len(weights))

    for i in range(training_size):
        for vector in tdm:
            distances = np.zeros(len(weights))
            euclidean_distance(clusters, distances, weights, vector)
            index = np.argmin(distances)                          # index of the best matching unit
            weights[index] = weights[index] + alpha * vector    # update weight of the best matching unit/ winning cluster

    # run distances to retrieve final answers
    for vector in tdm:
        distances = np.zeros(len(weights))
        euclidean_distance(clusters, distances, weights, vector)
        index = np.argmin(distances)                            # index of the best matching unit
        final.append(index)

    return final, weights


def normalize(list):
    sum = 0
    for i in list:
        sum += i ** 2
    res = math.sqrt(sum)
    return [list[0] / res, list[1] / res]


def main():
    training_size = 300
    alpha = 0.01
    clusters = [2, 3, 7]

    df = pd.read_csv("Ex1_data.txt", sep=',')
    vectors = df.values

    for i in range(len(clusters)):
        results, weights = wta_clustering(vectors, training_size, alpha, clusters[i])
        graph(weights, df, results)


if __name__ == '__main__':
    main()
