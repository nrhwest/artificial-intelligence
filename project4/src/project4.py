'''
Written by Yonathan Mekonnen, Derrick Adeji, Nathan West
CMSC 409
'''

import re
import sys
import string
import Porter_Stemmer_Python as PorterStemmer


def load_sentences():
    return re.sub(r"[^A-z \n]", "", open("sentences.txt", 'r').read().lower()).split('\n')


def load_stop_words():
    return [word for line in open("stop_words.txt", 'r') for word in line.split()]


def tokenize(sentence_list):
    return list(map(lambda sentence: sentence.lower().split(), sentence_list))


def remove_stop_words(sentence_list, stop_words):
    new_list = list()
    for line in sentence_list:
        new_list.append(list(filter(lambda word: not stop_words.__contains__(word), line)))
    return new_list


def main():

   sentences = load_sentences()
   stop_words = load_stop_words()

   sentence_list = tokenize(sentences)
   sentence_list = remove_stop_words(sentence_list, stop_words)

   print(sentence_list)

if __name__ == '__main__':
    main()
