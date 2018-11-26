'''
Written by Yonathan Mekonnen, Derrick Adeji, Nathan West
CMSC 409
11/25/18
'''

import re
import sys
import string
from collections import Counter
from Porter_Stemmer_Python import PorterStemmer


def tokenize(sentence_list):
    return list(map(lambda sentence: sentence.lower().split(), sentence_list))


def remove_stop_words(sentence_list, stop_words):
    new_list = list()
    for line in sentence_list:
        new_list.append(list(filter(lambda word: not stop_words.__contains__(word), line)))
    return new_list


def porter_stemming(sentence_list):
    stemmed_list = list()
    stemmer = PorterStemmer()
    for line in sentence_list:
        stemmed_list.append(list(map(lambda word: stemmer.stem(word, 0, len(word)-1), line)))

    return stemmed_list

def count_occurrences(sentence_list):
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
                tmp = list(occurrences)
                vector[tmp.index(word)] += 1      # increment the words occurrence from a given sentence?
        tdm.append(vector)

    return tdm


def main():
    sentences = re.sub(r"[^A-z \n]", "", open("sentences.txt", 'r').read().lower()).split('\n')
    stop_words = open("stop_words.txt", 'r').read().split('\n')

    sentence_list = tokenize(sentences)
    sentence_list = remove_stop_words(sentence_list, stop_words)

    stemmed_list = porter_stemming(sentence_list)
    occurrences = count_occurrences(stemmed_list)

    tdm = create_tdm(occurrences, stemmed_list)
    print(tdm)


if __name__ == '__main__':
    main()
