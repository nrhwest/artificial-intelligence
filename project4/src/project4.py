'''
Written by Yonathan Mekonnen, Derrick Adeji, Nathan West
CMSC 409
11/25/18
'''

import re
import sys
import string
from Porter_Stemmer_Python import PorterStemmer


def load_sentences():
    return re.sub(r"[^A-z \n]", "", open("sentences.txt", 'r').read().lower()).split('\n')


def load_stop_words():
    return open("stop_words.txt", 'r').read().split('\n')


def tokenize(sentence_list):
    return list(map(lambda sentence: sentence.lower().split(), sentence_list))


def remove_stop_words(sentence_list, stop_words):
    new_list = list()
    for line in sentence_list:
        new_list.append(list(filter(lambda word: not stop_words.__contains__(word), line)))
    return new_list


def stemming(sentence_list):
    new_list = list()
    stemmer = PorterStemmer()

    for line in sentence_list:
        new_list.append(list(map(lambda word: stemmer.stem(word, 0, len(word)-1), line)))

    return new_list

def main():
    sentences = load_sentences()
    stop_words = load_stop_words()

    sentence_list = tokenize(sentences)
    sentence_list = remove_stop_words(sentence_list, stop_words)

    stemmed_sentence_list = stemming(sentence_list)
    print(stemmed_sentence_list)

    # print(sentence_list)

if __name__ == '__main__':
    main()
