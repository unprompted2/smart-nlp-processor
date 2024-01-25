#!/usr/bin/env python3

import msgpack
import msgpack_numpy
import sys
import numpy as np

print("Loading most common words in english language")
common_words = sys.argv[1] + '/most_common_words.txt'

# read as txt file
with open(common_words, 'r') as common_words_file:
    content = common_words_file.readlines()
    common_words_l = [x.strip() for x in content]

print("read {} words from common words file".format(len(common_words_l)))

print("loading serialized big_dictionary")

with open(sys.argv[1] + '/big_dictionary', 'rb') as big_dictionary_file:
    big_dictionary = msgpack.load(big_dictionary_file, raw=False)

print("loading serialized big_wordvectors")

with open(sys.argv[1] + '/big_wordvectors', 'rb') as big_wordvectors_file:
    big_wordvectors = msgpack_numpy.load(big_wordvectors_file)

print("Finding most common words in big_dictionary and generating reduced size dictionary")
print("--------")

i = 0
dictionary = {}
wordvectors = []

number_of_loops = len(common_words_l)

for common_word in common_words_l:
    # get index
    try:
        index = big_dictionary[common_word]
        # from the index we get the correspinding vector
        # generate a subset of big big_dictionary called dictionary
        # the following line does : d