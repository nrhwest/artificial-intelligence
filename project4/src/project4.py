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

    return tdm


def print_to_csv(tdm):  # not sure if this is correct
    with open("term_document_matrix.csv", 'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(tdm)


def wta_clustering(tdm, stemmed_list, training_size, alpha):
    count = 5
    weights = np.random.rand(5, len(tdm[0]))

    # print(weights)
    # print(weights[0])
    for i in range(training_size):

        for vector in tdm:
            distances = np.zeros(len(weights))

            for x in range(len(weights)):
                distances[x] = 0

                for y in range(len(vector)):
                    val = pow((vector[y] - weights[x][y]), 2)
                    distances[x] += val

            index = -1
            for x in range(len(distances)):
                if distances[x] < distances[index]:
                    index = i
            print(index)
# def normalization(tdm):



def main():

    training_size = 500
    alpha = 0.3

    sentences = re.sub(r"[^A-z \n]", "", open("sentences.txt", 'r').read().lower()).split('\n')
    stop_words = open("stop_words.txt", 'r').read().split('\n')

    sentence_list = tokenize(sentences)
    sentence_list = remove_stop_words(sentence_list, stop_words)

    stemmed_list = porter_stemming(sentence_list)
    occurrences = combined_stemmed_words(stemmed_list)

    tdm = list(occurrences) + create_tdm(occurrences, stemmed_list)
    # print_to_csv(tdm)

    tdm2 = create_tdm(occurrences, stemmed_list)
    wta_clustering(tdm2, stemmed_list, training_size, alpha)

if __name__ == '__main__':
    main()
