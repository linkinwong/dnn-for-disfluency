__author__ = 'brtdra'

from network import *

from corpus_processing import *


def main():
    train_set, test_set = corpus_processing('ssr.tr.sents', 'ssr.te.sents', 'ssr.tr.annotated.bin',
                                            'ssr.te.annotated.bin')

    run_network(train_set, test_set, 'prova')


if __name__ == '__main__':
    main()
