
'''
Written by Yonathan Mekonnen, Derrick Adeji, Nathan West
CMSC 409
11/25/18
'''

import re
import sys
import csv
import string
import random
from collections import Counter
import numpy as np
from Porter_Stemmer_Python import PorterStemmer


def tokenize(sentence_list):                      # return a list of tokens within another list
    return list(map(lambda sentence: sentence.lower().split(), sentence_list))


def remove_stop_words(sentence_list, stop_words):
    new_sentence_list = list()

    for line in sentence_list:                    # remove/filter out stop_words from the sentence add to new list
        new_sentence_list.append(list(filter(lambda word: not stop_words.__contains__(word), line)))

    return new_sentence_list


def porter_stemming(sentence_list):
    stemmed_list = list()
    stemmer = PorterStemmer()                     # instantiate porter stemmer algorithm

    for line in sentence_list:                    # loop through sentence_list, stem each word, add to new list
        stemmed_list.append(list(map(lambda word: stemmer.stem(word, 0, len(word)-1), line)))

    file = open("feature_vectors.csv", "w")
    writer = csv.writer(file)
    writer.writerows(stemmed_list)

    return stemmed_list

def combined_stemmed_words(sentence_list):
    count = dict()

    for sentence in sentence_list:
        for word in sentence:
            if word in count:
                count[word] += 1
            else:
                count[word] = 1

    return count


def create_tdm(occurrences, stemmed_list):
    tdm = list()

    for sentence in stemmed_list:                 # each sentence (list) in stemmed_list
        vector = [0] * len(occurrences)           # make freq_vector the size of length of occurrences

        for word in sentence:                     # each token in list
            if (word in occurrences):
                tmp = list(occurrences)           # create a list of the stemmed words dictionary
                vector[tmp.index(word)] += 1      # increment the words occurrence from a given sentence?

        tdm.append(vector)
        tdm_file = open("tdm.csv", 'w')
        writer = csv.writer(tdm_file)
        writer.writerows(tdm)

    return tdm


def euclidean_distance(clusters, distances, weights, vector):
    for i in range(clusters):
        distances[i] = 0
        for j in range(len(vector)):
            val = (weights[i][j] - vector[j]) ** 2
            distances[i] += val

<<<<<<< HEAD
=======
def euclidean_distance(clusters, distances, weights, vector):
    for i in range(clusters):
        distances[i] = 0
        for j in range(len(vector)):
            val = (weights[i][j] - vector[j]) ** 2
            distances[i] += val
>>>>>>> 00adcc572f4fc57adfee172c64e12993fcda8568

def wta_clustering(tdm, stemmed_list, training_size, alpha, clusters):
    weights = np.random.rand(clusters, len(tdm[0]))

    final = list()
    distances = np.zeros(len(weights))
    tdm = np.array(tdm)
<<<<<<< HEAD

    for i in range(training_size):
        for vector in tdm:
            distances = np.zeros(len(weights))
            euclidean_distance(clusters, distances, weights, vector)
            index = np.argmin(distances)                          # index of the best matching unit
            weights[index] = weights[index] + alpha * vector    # update weight of the best matching unit/ winning cluster

=======
    for i in range(training_size):
        distances = np.zeros(len(weights))
        for vector in tdm:
            euclidean_distance(clusters, distances, weights, vector)
            idx = np.argmin(distances)                      #index of the best matching unit
            weights[idx] = weights[idx] + alpha * vector    #update weight of the best matching unit/ winning cluster
>>>>>>> 00adcc572f4fc57adfee172c64e12993fcda8568
    # run distances to retrieve final answers
    for vector in tdm:
        distances = np.zeros(len(weights))
        euclidean_distance(clusters, distances, weights, vector)
<<<<<<< HEAD
        index = np.argmin(distances)                            # index of the best matching unit
        final.append(index)
=======
        idx = np.argmin(distances)  #index of the best matching unit
        final.append(idx)
    return final
>>>>>>> 00adcc572f4fc57adfee172c64e12993fcda8568

    return final


def main():
    training_size = 1000
    alpha = 0.3
    num_clusters = 8
<<<<<<< HEAD

=======
>>>>>>> 00adcc572f4fc57adfee172c64e12993fcda8568
    sentences = re.sub(r"[^A-z \n]", "", open("sentences.txt", 'r').read().lower()).split('\n')
    stop_words = open("stop_words.txt", 'r').read().split('\n')

    sentence_list = tokenize(sentences)
    sentence_list = remove_stop_words(sentence_list, stop_words)

    stemmed_list = porter_stemming(sentence_list)
    occurrences = combined_stemmed_words(stemmed_list)

    tdm = list(occurrences) + create_tdm(occurrences, stemmed_list)
    tdm2 = create_tdm(occurrences, stemmed_list)
<<<<<<< HEAD

    results = wta_clustering(tdm2, stemmed_list, training_size, alpha, clusters=num_clusters)

    original_sentences = open("sentences.txt", 'r').read().split('\n')

=======
    results = wta_clustering(tdm2, stemmed_list, training_size, alpha, clusters=num_clusters)
    original_sentences = open("sentences.txt", 'r').read().split('\n')
    print(results)
>>>>>>> 00adcc572f4fc57adfee172c64e12993fcda8568
    for i in range(num_clusters):
        print("Cluster {}: ".format(i))
        print("------------")
        for x in range(len(results)):
            if results[x] == i:
                print(original_sentences[x])


if __name__ == '__main__':
    main()
