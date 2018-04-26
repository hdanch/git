#!/usr/bin/env python3
import sys
sys.path.append("../")
import utils
import settings
import logging
import numpy as np
from hcfeatures import futils
from senticnet4 import senticnet

LOGGER = logging.getLogger("task3")

tokenizer = utils.tokenizer

opinion_pos_dict = futils.opinion_pos_dict

opinion_neg_dict = futils.opinion_neg_dict

def get_sentiment_features(tweet):
    x = np.array([])

    # split words with nltk tweet tokenizer
    words = tokenizer(tweet[0].lower())

    # opinion positive and negative
    #x = np.concatenate((x, get_opinion_sentiment(words)), axis=0)

    #afinn 96 and 111
    #x = np.concatenate((x, get_afinn_sentiment(words)), axis=0)

    #senticnet
    x = np.concatenate((x, get_senticnet_sentiment(words)), axis=0)

    return x.flatten()

def get_afinn_sentiment(words):
    afinn_111_score = sum(map(lambda word: futils.afinn_111_dict.get(word, 0), words))
    afinn_96_score = sum(map(lambda word: futils.afinn_96_dict.get(word, 0), words))
    afinn = []
    #afinn.append(afinn_111_score)
    #afinn.append(afinn_96_score)
    #afinn.append(afinn_111_score + afinn_96_score)
    return afinn

def get_opinion_sentiment(words):
    pos_score = sum(map(lambda word: futils.opinion_pos_dict.get(word, 0), words))
    neg_score = sum(map(lambda word: futils.opinion_neg_dict.get(word, 0), words))
    opinion = []
    #opinion.append(pos_score)
    #opinion.append(neg_score)
    #opinion.append(pos_score + neg_score)
    #opinion.append(pos_score / len(words))
    return opinion

def get_senticnet_sentiment(words):
    one_score = sum(map(lambda word: float(senticnet.get(word, [0])[0]), words))
    two_score = sum(map(lambda word: float(senticnet.get(word, [0]*2)[1]), words))
    three_score = sum(map(lambda word: float(senticnet.get(word, [0]*3)[2]), words))
    four_score = sum(map(lambda word: float(senticnet.get(word, [0]*4)[3]), words))
    eight_score = sum(map(lambda word: float(senticnet.get(word, [0]*8)[7]), words))
    net = []
    #net.append(one_score)
    #net.append(one_score / len(words))
    net.append(two_score)
    #net.append(two_score / len(words))
    #net.append(three_score)
    #net.append(three_score / len(words))
    #net.append(four_score)
    #net.append(four_score / len(words))
    #net.append(eight_score)
    #net.append(eight_score / len(words))
    #net.append(len(words))
    return net



