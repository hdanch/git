#!/usr/bin/env python3
import sys
sys.path.append("../")
import utils
import settings
import logging
import numpy as np
import utils

LOGGER = logging.getLogger("task3")

afinn_111_dict = utils.read_key_int('corpus/AFINN/AFINN-111.txt')
LOGGER.debug('Length of afinn 111: ' + str(len(afinn_111_dict)))

afinn_96_dict = utils.read_key_int('corpus/AFINN/AFINN-96.txt')
LOGGER.debug('Length of afinn 96: ' + str(len(afinn_96_dict)))

opinion_pos_dict = utils.read_word_list('corpus/opinion-lexicon-English/positive-words.txt', escape_char=';', isToDict=True, value=1)
LOGGER.debug('Length of opinion pos: ' + str(len(opinion_pos_dict)))

opinion_neg_dict = utils.read_word_list('corpus/opinion-lexicon-English/negative-words.txt', escape_char=';', isToDict=True, value=-1)
LOGGER.debug('Length of opinion neg: ' + str(len(opinion_neg_dict)))


