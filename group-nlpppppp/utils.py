#!/usr/bin/env python3
#utils
import numpy as np
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from hcfeatures import *
import constant
import settings
import logging

LOGGER = logging.getLogger("task3")

LOGGER.info('Init tokenizer...')
tokenizer = TweetTokenizer(preserve_case=True,
                           reduce_len=False,
                           strip_handles=False).tokenize

LOGGER.info('Init vectorizer...' + constant.TFIDF_VEC)
vectorizer = joblib.load(constant.TFIDF_VEC)
'''
vectorizer = TfidfVectorizer(lowercase=True, 
                             strip_accents="unicode", 
                             analyzer="word", 
                             tokenizer=tokenizer, 
                             stop_words="english", 
                             vocabulary=vocabulary)
'''

def parse_dataset(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    y = []
    corpus = []
    with open(fp, 'rt') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                label = int(line.split("\t")[1])
                tweet = line.split("\t")[2]
                y.append(label)
                corpus.append(tweet)

    return corpus, y

def parse_tweet(isTest=False):
    ark_file = './data/arknlp' + str('_test.txt' if isTest else '.txt')
    tw_file = './data/twnlp' + str('_test.txt' if isTest else '.txt')

    print(ark_file, tw_file)
    ark_words_list = list()
    ark_pos_list = list()
    with open(ark_file, 'r') as f:
        for line in f.readlines():
            raw = line.strip().split('\t')
            ark_words_list.append(raw[0].strip())
            ark_pos_list.append(raw[1].strip())

    tw_words_list = []
    tw_pos_list = []
    tw_entity_list = []
    with open(tw_file, 'r') as f:
        for line in f.readlines():
            words = []
            pos = []
            entity =[]
            raw = line.strip().split(' ')
            for w in raw:
                w = w.strip().split('/')
                pos.append(w[-1])
                entity.append(w[-2])
                if len(w) > 3:
                    words.append('/'.join(w[:-2]))
                else:
                    words.append(w[0])
            tw_words_list.append(words)
            tw_pos_list.append(pos)
            tw_entity_list.append(entity)
    return (ark_words_list, ark_pos_list, tw_words_list, tw_pos_list, tw_entity_list)

def replaceURLAndAt(line):
    pass

def read_key_int(filepath, split_regex='\t' , escape_char='#'):
    kv_dict = dict()
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line.startswith(escape_char):
                raw = line.split(split_regex)
                if len(raw) == 2:
                    kv_dict[raw[0]] = int(raw[1])
    return kv_dict

def read_word_list(filepath, escape_char='#', isToDict=False, value=0):
    words = list()
    kv_dict = dict()
    with open(filepath, 'r', encoding='iso-8859-1') as f:
        for line in f.readlines():
            line = line.strip()
            if not line.startswith(escape_char):
                words.append(line)
                kv_dict[line] = value
    if isToDict:
        return kv_dict
    return words

