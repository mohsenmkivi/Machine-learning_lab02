import numpy as np
import collections
import os

h = open("aclImdb/aclImdb/stopwords.txt", "r")
list = h.read()
common_words = list.split()

def not_common_word(word):
    """Define if a word is common or uncommon in respect to the stopwords.txt file"""
    n = 0
    for j in range(319):
        if word.lower() == l[j]:
            n += 1
    #print(word, n, n==0)
    if n == 0:
        return True
    else:
        return False

"""
def load_stopwords(filename):
    Load stopwords from a file.
    with open(filename) as f:
        stopwords = set(f.read().split())
    return stopwords

print("common_words")
"""