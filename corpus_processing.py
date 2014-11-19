__author__ = 'brtdra'

import numpy as np
import scipy.sparse as sp


class Word(object):
    def __init__(self, word, pos, label):
        self.word = sp.lil_matrix(word)
        self.pos = sp.lil_matrix(pos)
        self.label = label


class Sentence(object):
    def __init__(self, word_list):
        self.words = word_list

    def getLabelsArray(self):
        ret = np.zeros(len(self.words))

        for i in range(len(self.words)):
            ret[i] = self.words[i].label

        return ret

    def getTotalInputLength(self):
        return self.words[0].word.shape[1] + self.words[0].pos.shape[1]

    def getInputArray(self):
        ret = np.zeros((len(self.words), self.getTotalInputLength()))

        for i in range(len(self.words)):
            ret[i] = np.concatenate((self.words[i].word.todense(), self.words[i].pos.todense()), 1)

        return ret


def find_all_words(all_lines):
    all_words = set()
    all_pos = set()
    all_labels = set()

    # Sentence boundaries
    all_words.add('<s>')
    all_pos.add('<s>')
    all_words.add('</s>')
    all_pos.add('</s>')

    for l in all_lines:
        if len(l.strip()) > 0:
            fields = l.split(' ')

            all_words.add(fields[0])
            all_pos.add(fields[1])
            all_labels.add(fields[2])

    return sorted(list(all_words)), sorted(list(all_pos)), sorted(list(all_labels))


def corpus_processing(train_fname, test_fname):
    def start_new_sentence(word_list):
        sentence = []
        word_array = np.zeros(len(word_list[0]) + 1)
        pos_array = np.zeros(len(word_list[1]) + 1)
        wix = word_list[0].index('<s>')
        pix = word_list[1].index('<s>')
        word_array[wix] = 1
        pos_array[pix] = 1
        sentence.append(Word(word_array, pos_array, -1))

        return sentence


    trainf = open(train_fname)
    train_lines = trainf.readlines()
    trainf.close()

    word_list = find_all_words(train_lines)

    train_corpus = []

    j = 0

    sentence = start_new_sentence(word_list)
    for line in train_lines:
        if len(line.strip()) > 0:
            line_fields = line.split(' ')
            word_array = np.zeros(len(word_list[0]) + 1)
            pos_array = np.zeros(len(word_list[1]) + 1)

            if line_fields[0] in word_list[0]:
                ix = word_list[0].index(line_fields[0])
                word_array[ix] = 1

            else:
                word_array[-1] = 1

            if line_fields[1] in word_list[1]:
                ix = word_list[1].index(line_fields[1])
                pos_array[ix] = 1

            else:
                pos_array[-1] = 1

            lbl = -1 if line_fields[2].strip() == 'O' else 1

            sentence.append(Word(word_array, pos_array, lbl))

        else:
            word_array = np.zeros(len(word_list[0]) + 1)
            pos_array = np.zeros(len(word_list[1]) + 1)
            wix = word_list[0].index('</s>')
            pix = word_list[1].index('</s>')
            word_array[wix] = 1
            pos_array[pix] = 1
            sentence.append(Word(word_array, pos_array, -1))
            train_corpus.append(Sentence(sentence))
            sentence = start_new_sentence(word_list)


            print 'Processed train sentence ' + str(j)
            j += 1


    testf = open(test_fname)
    test_corpus = []

    j = 0

    sentence = start_new_sentence(word_list)
    for line in testf:
        if len(line.strip()) > 0:
            line_fields = line.split(' ')
            word_array = np.zeros(len(word_list[0]) + 1)
            pos_array = np.zeros(len(word_list[1]) + 1)

            if line_fields[0] in word_list[0]:
                ix = word_list[0].index(line_fields[0])
                word_array[ix] = 1

            else:
                word_array[-1] = 1

            if line_fields[1] in word_list[1]:
                ix = word_list[1].index(line_fields[1])
                pos_array[ix] = 1

            else:
                pos_array[-1] = 1

            lbl = -1 if line_fields[2].strip() == 'O' else 1

            sentence.append(Word(word_array, pos_array, lbl))

        else:
            word_array = np.zeros(len(word_list[0]) + 1)
            pos_array = np.zeros(len(word_list[1]) + 1)
            wix = word_list[0].index('</s>')
            pix = word_list[1].index('</s>')
            word_array[wix] = 1
            pos_array[pix] = 1
            sentence.append(Word(word_array, pos_array, -1))
            test_corpus.append(Sentence(sentence))
            sentence = start_new_sentence(word_list)

            print 'Processed test sentence ' + str(j)
            j += 1


    testf.close()

    return train_corpus, test_corpus
