__author__ = 'brtdra'

from network import *

from corpus_processing import *


def main():
    train_set, test_set = corpus_processing('train.txt', 'test.txt')

    print test_set


if __name__ == '__main__':
    main()
