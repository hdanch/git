#!/usr/bin/env python3
import sys
from sklearn.feature_extraction.text import CountVectorizer 
sys.path.append("../")
import utils
import settings
import logging
import numpy as np
import re


LOGGER = logging.getLogger("task3")

tokenizer = utils.tokenizer

vectorizer = utils.vectorizer

def get_lexical_features(tweet):
    x = np.array([])

    #add your feature list here
    x = np.concatenate((x, get_test(tweet)), axis=0)

    return x.flatten()

pattern = r'(.{1,4})\1\1'
def get_test(tweet):
    test_vec = []

    words = tokenizer(tweet[0].lower())
    flag = 0
    for word in words:
        if word.isupper() and len(word) > 3:
            flag += 1

    count_exclamation = tweet[0].count('!')
    count_dot = tweet[0].count('.')
    count_hashtag = tweet[0].count('#')

    #test_vec.append(count_dot)
    #test_vec.append(count_exclamation)
    test_vec.append(flag)
    #test_vec.append(1 if tweet[0].count('#') > 5 else 0)
    #test_vec.append(1 if count_hashtag > 5 else 0)
    flooding_set = set(re.findall(pattern, tweet[0].lower()))
    pun_flooding = 0
    char_flooding = 0
    char_num = 0
    for s in flooding_set:
        if re.match('\W+', s):
            pun_flooding = 1
        elif re.match('\w+', s):
            char_flooding = 1
        if len(s) > char_num:
            char_num = len(s)
    #test_vec.append(0 if len(flooding_set) == 0 else 1)
    #test_vec.append(pun_flooding)
    #test_vec.append(char_flooding)
    #test_vec.append(char_num)


    '''
    char_vectorizer = CountVectorizer(analyzer='char', strip_accents="unicode", ngram_range=(1,4), stop_words=[' '])
    count_repeat = char_vectorizer.fit_transform([tweet[0].replace(' ', '')])
    print(char_vectorizer.get_feature_names())
    print(tweet[0])
    flooding = 0
    for (c, n) in zip(count_repeat.toarray()[0], char_vectorizer.get_feature_names()):
        if (len(n) == 1 and c >=10) or (len(n) == 2 and c >=3 ) or (len(n) ==3 and c >= 3 ) or (len(n) == 4 and c >= 3):
            print(n + ': ' + str(c))
            flooding += 1
    '''
    return test_vec


