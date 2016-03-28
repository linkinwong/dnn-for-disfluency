__author__ = 'brtdra'

from network import *

from corpus_processing import *

import logging

logger = logging.getLogger(__name__)

def main():
    train_set, development_set, test_set = corpus_processing('train.txt', 'development.txt', 'test.txt')

    #run_network(train_set, development_set, test_set, 'prova_uno_')


if __name__ == '__main__':
    log_file = "log_file.txt"
    logging.basicConfig(filename=log_file,filemode='w', level = logging.WARNING)
    main()
