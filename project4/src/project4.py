'''
Written by Yonathan Mekonnen, Derrick Adeji, Nathan West
CMSC 409
11/25/18
'''

import re
import sys
import string
from collections import Counter

# import textmining
# from textmining.stemmer import textmining
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
    stemmed_list = list()
    stemmer = PorterStemmer()
    for line in sentence_list:
        stemmed_list.append(list(map(lambda word: stemmer.stem(word, 0, len(word)-1), line)))

    return stemmed_list

def frequency_vector(sentence_list):
    count = dict()
    for sentence in sentence_list:
        for word in sentence:
            if word in count:
                count[word] += 1
            else:
                count[word] = 1
    return count


def create_tdm(freq_vector, sentence_list):
    # print(freq_vector)
    print(len(freq_vector))
    # tdm = freq_vector.copy()
    tdm = dict.fromkeys(freq_vector, 0)
    # print(tdm)
    # print(tdm)
    # exit()
    # print(sentence_list[0])
    # tdm = [0] * (len(freq_vector))
    #for w in sentence_list[0]:
    #    tdm.append(sentence_list[0](w))
    #    print(str(tdm)
    # for key in freq_vector:
    #     for j in range(len(sentence_list)):
    #         # print(sentence_list[j])
    #         if (sentence_list[j][0] == key):
    #             # print(sentence_list[j][0])
    #             # counts = Counter(sentence_list[i])
    #             tdm[[key][j]] += 1
    #             # tdm.append(sentence_list[j].count(freq_vector(i)))
    #             # print(str(tdm))
    #             print(tdm)
    #             break
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
    sentences = load_sentences()
    stop_words = load_stop_words()

    sentence_list = tokenize(sentences)
    sentence_list = remove_stop_words(sentence_list, stop_words)

    stemmed_sentence_list = stemming(sentence_list)
    print(len(stemmed_sentence_list))
    # counts = Counter(stemmed_sentence_list[0])
    # print(counts)
    freq_vector = frequency_vector(stemmed_sentence_list)

    create_tdm(freq_vector, stemmed_sentence_list)


if __name__ == '__main__':
    main()
