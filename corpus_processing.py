__author__ = 'brtdra'

import numpy as np


class Sentence(object):
    def __init__(self, array, label):
        self.array = array
        self.label = label


def find_all_words(all_lines):
    ret = dict()

    for l in all_lines:
        words = l.split(' ')

        for w in words:
            ret[w] = 1

    return sorted(list(ret.keys()))


def corpus_processing(train_fname, test_fname, train_lb_fname, test_lb_fname):
    trainf = open(train_fname)
    train_lines = trainf.readlines()
    trainf.close()

    word_list = find_all_words(train_lines)

    train_corpus = []

    trainlbf = open(train_lb_fname)

    j = 0

    for line, label in zip(train_lines, trainlbf):
        line_toks = line.split(' ')
        a = np.zeros((len(line_toks), len(word_list) + 1))

        for i in np.arange(len(line_toks)):
            ix = word_list.index(line_toks[i])

            if ix == -1:
                ix = 0

            else:
                ix += 1

            a[i, ix] = 1

        if label == 0:
            label = -1

        train_corpus.append(Sentence(a, label))
        print 'Processed sentence ' + str(j)
        j += 1

    trainlbf.close()

    testf = open(test_fname)
    testlbf = open(test_lb_fname)

    test_corpus = []
    j = 0

    for line, label in zip(testf, testlbf):
        line_toks = line.split(' ')
        a = np.zeros((len(line_toks), len(word_list) + 1))

        for i in np.arange(len(line_toks)):
            try:
                ix = word_list.index(line_toks[i]) + 1

            except ValueError:
                ix = 0

            a[i, ix] = 1

        if label == 0:
            label = -1

        test_corpus.append(Sentence(a, label))
        print 'Processed sentence ' + str(j)
        j += 1

    testf.close()
    testlbf.close()

    return train_corpus, test_corpus
