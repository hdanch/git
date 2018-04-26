#!/usr/bin/env python3
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import dump_svmlight_file
from sklearn import metrics
import numpy as np
import sys
import logging

import settings
import utils
import features

LOGGER = logging.getLogger("task3")

def main(task='A'):
    TASK = task
    DATASET_FP = './data/SemEval2018-T4-train-task' + TASK + '.txt'
    FNAME = './result/predictions-task' + TASK + '.txt'
    LOGGER.debug('Task file ' + DATASET_FP)
    LOGGER.debug('Result file ' + FNAME)

    PREDICTIONSFILE = open(FNAME, "w")

    K_FOLDS = 10 # 10-fold crossvalidation
    CLF = LinearSVC(tol=1e-8) # the default, non-parameter optimized linear-kernel SVM

    # Loading dataset and featurised simple Tfidf-BoW model
    corpus, y = utils.parse_dataset(DATASET_FP)

    # Loading tokenized tweet, pos and entity from ark and tw
    syn_info = utils.parse_tweet()

    # Get features
    X = features.featurize(corpus, syn_info)

    class_counts = np.asarray(np.unique(y, return_counts=True)).T.tolist()
    LOGGER.debug('class counts ' + str(class_counts))

    # Returns an array of the same size as 'y' where each entry is a prediction obtained by cross validated
    predicted = cross_val_predict(CLF, X, y, cv=K_FOLDS)

    # confusion matrix
    confu = metrics.confusion_matrix(y, predicted)
    print(confu)
    # Modify F1-score calculation depending on the task
    if TASK.lower() == 'a':
        score = metrics.f1_score(y, predicted, pos_label=1)
    elif TASK.lower() == 'b':
        score = metrics.f1_score(y, predicted, average="macro")
    LOGGER.debug("F1-score Task" + TASK + ': ' + str(score*100))
    print('**********************************')
    print ("F1-score Task", TASK, score*100)
    print('**********************************')
    for p in predicted:
        PREDICTIONSFILE.write("{}\n".format(p))
    PREDICTIONSFILE.close()

if __name__ == '__main__':
    TASK = sys.argv[1]
    main(task=TASK)

