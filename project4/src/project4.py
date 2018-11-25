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
    return list(map(lambda sentence: sentence.lower().split(), sentences))

def remove_stop_words(sentence_list, stop_words):
    for line in sentence_list:
        for word in line:
            if word in stop_words:
                print("before removal, line = ", line)
                print("WORD = ", word)
                line.remove(word)
                print("after removal, line = ", line)
                print()
                # break
        # break
    return sentence_list

sentences = load_sentences()
stop_words = load_stop_words()
# print(stop_words)

sentence_list = tokenize(sentences)
sentence_list = remove_stop_words(sentence_list, stop_words)

print(sentence_list)
