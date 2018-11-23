'''
Written by Yonathan Mekonnen, Derrick Adeji, Nathan West
CMSC 409
'''

import re
import sys
import string

sentences = open("sentences.txt", mode='r')
stop_words = open("stop_words.txt", mode='r')

# make a 2D list of sentences
sentences_list = list(map(lambda sentence: sentence.lower().split(), sentences))
print(sentences_list)

for lines in sentences_list:
    for word in lines:
        if re.match(r'[0-9]*', word):
            ind = lines.index(word)
            del[ind]

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
