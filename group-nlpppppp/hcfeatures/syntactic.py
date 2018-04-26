#!/usr/bin/env python3
import sys
sys.path.append("../")
import utils
import settings
import logging
import numpy as np

LOGGER = logging.getLogger("task3")

tokenizer = utils.tokenizer

vectorizer = utils.vectorizer

def get_syntactic_features(tweet):
    x = np.array([])

    #add your feature list here
    #x = np.concatenate((x, your_feature_list), axis=0)

    return x.flatten()
