'''
Written by Yonathan Mekonnen, Derrick Adeji, Nathan West
CMSC 409
'''

import re
import sys
import string
import Porter_Stemmer_Python as PorterStemmer

sentences = open("sentences.txt", mode='r')
stop_words = open("stop_words.txt", mode='r')

# make a 2D list of sentences
sentences_list = list(map(lambda sentence: sentence.lower().split(), sentences))
# print(sentences_list)'

for lines in sentences_list:
    for word in lines:
        # if word.find()
        if re.match(r'[^A-Za-z0-9]+', word):
            re.sub(r'[^A-Za-z0-9]+', '', word)
        if re.match(r'\d+', word):
            ind = lines.index(word)
            del lines[ind]

# porter = PorterStemmer()

print(sentences_list)
# the_list = list(map(lambda word: word.translate(string.punctuation), sentences_list))
# all_tokens = []
# for line in sentences:
#     uniques = set(line.split())
#     freqs = [(item, line.split(" ")) for item in uniques]
#     print(freqs)
#     print("\n\n")
    # print(line, sep=' ', en   d='n', file=sys.stdout, flush=False)
    # print(string.punctuation)

# print(all_tokens)
