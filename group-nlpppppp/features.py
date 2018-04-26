#!/usr/bin/env python3
import numpy as np
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.externals import joblib
from hcfeatures import lexical, sentiment, syntactic, semantic, others
import constant
import settings
import logging
import utils
from sklearn.externals import joblib

LOGGER = logging.getLogger("task3")

space_tokenizer = lambda x:x.split(' ')
nltk_tokenizer = utils.tokenizer
#nltk_vectorizer = utils.vectorizer

def featurize(corpus, syn_info, isTest=False):
    X = None

    # nltk TF-IDF
    nltk_vec = TfidfVectorizer(lowercase=False, strip_accents="unicode", analyzer="word", tokenizer=nltk_tokenizer, stop_words="english", ngram_range=(1,4))
    X = nltk_vec.fit_transform(corpus)
    X = X.toarray()
    LOGGER.debug('nltk_vec shape ' + str(X.shape))

    # ark TF-IDF
    ark_vectorizer = TfidfVectorizer(lowercase=False, strip_accents="unicode", analyzer="word", tokenizer=space_tokenizer, stop_words="english")#, ngram_range=(2,2))
    ark_vec = ark_vectorizer.fit_transform(syn_info[0])
    #X = np.concatenate((X, ark_vec.toarray()), axis=1)
    LOGGER.debug('ark_vec shape ' + str(ark_vec.shape))

    # tw TF-IDF
    tw_words_list = map(lambda x:' '.join(x), syn_info[2])
    tw_vectorizer = TfidfVectorizer(lowercase=False, strip_accents="unicode", analyzer="word", tokenizer=space_tokenizer, stop_words="english")#, ngram_range=(1,2))
    tw_vec = tw_vectorizer.fit_transform(tw_words_list)
    #X = np.concatenate((X, tw_vec.toarray()), axis=1)
    LOGGER.debug('tw_vec shape ' + str(tw_vec.shape))

    # ark pos count
    ark_pos_vectorizer = CountVectorizer(tokenizer=space_tokenizer, ngram_range=(1,2))
    ark_pos_vec = ark_pos_vectorizer.fit_transform(syn_info[1])
    LOGGER.debug('ark_pos_vec shape ' + str(ark_pos_vec.shape))
    X = np.concatenate((X, ark_pos_vec.toarray()), axis=1)

    # tw pos count
    tw_pos_list = map(lambda x:' '.join(x), syn_info[3])
    tw_pos_vectorizer = CountVectorizer(tokenizer=space_tokenizer, ngram_range=(1,2))
    tw_pos_vec = tw_pos_vectorizer.fit_transform(tw_pos_list)
    LOGGER.debug('tw_pos_vec shape ' + str(tw_pos_vec.shape))
    #X = np.concatenate((X, tw_pos_vec.toarray()), axis=1)

    # get hand-craft features
    hc_X = []
    for (tweet, ark_ws, ark_ps, tw_ws, tw_ps) in zip(corpus, syn_info[0], syn_info[1], syn_info[2], syn_info[3]):
        tweet_info = (tweet, ark_ws, ark_ps, tw_ws, tw_ps)
        x = featurize_one(tweet_info)
        hc_X.append(x)
    hc_X = np.asarray(hc_X)
    LOGGER.debug('hc_X shape ' + str(hc_X.shape))

    X = np.concatenate((X, hc_X), axis=1)
    LOGGER.debug('X shape ' + str(X.shape))

    return X

'''
featurize one tweet
input:
    tweet_info:
        tweet: original tweet
        ark_ws: ark tokenized words
        ark_ps: ark pos
        tw_ws: tw tokenized words
        tw_ps: tw pos
'''
def featurize_one(tweet_info):
    return get_handcraft_features(tweet_info)

def get_handcraft_features(tweet):
    x = np.array([])
    x = np.concatenate((x, lexical.get_lexical_features(tweet)), axis=0)
    x = np.concatenate((x, syntactic.get_syntactic_features(tweet)), axis=0)
    x = np.concatenate((x, sentiment.get_sentiment_features(tweet)), axis=0)
    x = np.concatenate((x, semantic.get_semantic_features(tweet)), axis=0)
    x = np.concatenate((x, others.get_other_features(tweet)), axis=0)
    return x


