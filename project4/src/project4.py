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


def stemming(sentence_list):
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


def create_tdm(freq_vector, sentence_list):
    # tdm = freq_vector.copy()
    tdm = dict.fromkeys(freq_vector, 0)
    for j in range(len(sentence_list)):
        for key in freq_vector:
            # print(sentence_list[j])
            # print(sentence_list[])
            if (sentence_list[j][0] == key):
                # print(sentence_list[j][0])
                # counts = Counter(sentence_list[i])
                tdm[[j][key]] += 1
                # tdm.append(sentence_list[j].count(freq_vector(i)))
                # print(str(tdm))
                print(tdm)
                break

def main():
    sentences = re.sub(r"[^A-z \n]", "", open("sentences.txt", 'r').read().lower()).split('\n')
    stop_words = open("stop_words.txt", 'r').read().split('\n')

    sentence_list = tokenize(sentences)
    sentence_list = remove_stop_words(sentence_list, stop_words)

    stemmed_list = stemming(sentence_list)
    occurrences = count_occurrences(stemmed_list)

    # create_tdm(freq_vector, stemmed_sentence_list)

    # not right - but getting close
    all_records = list()
    for sentence in stemmed_list:                   # each sentence (list) in stemmed_list
        vector = [0] * len(occurrences)             # make freq_vector the size of length of occurrences
        for word in sentence:                       # each token in list
            # if (freq_vector.__contains__(word)):
            if (word in occurrences):
                vector[occurrences.get(word)] += 1  # increment the words occurrence from a given sentence?
        all_records.append(vector)

    print(all_records)


if __name__ == '__main__':
    main()
