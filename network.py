__author__ = 'brtdra'

from accuracy_plot import *

import numpy as np
import theano.tensor as T
import theano
import theano.tensor.nnet as nnet
import math


class EmbeddingLayer(object):
    def __init__(self, input, input_size, output_size):
        We_val = np.asarray(np.random.uniform(
            low=-np.sqrt(6.0 / (input_size + output_size * 2)),
            high=np.sqrt(6.0 / (input_size + output_size * 2)),
            size=(input_size, output_size)), dtype=theano.config.floatX)

        self.We = theano.shared(We_val, 'We', borrow=True)

        be_val = np.zeros((output_size,), dtype=theano.config.floatX)
        self.be = theano.shared(value=be_val, name='be', borrow=True)

        self.output = T.tanh(T.dot(input, self.We) + self.be)
        self.param = [self.We, self.be]


class EncoderLayer(object):
    def __init__(self, input, input_size, rnn_inner_size):
        W_val = np.asarray(np.random.uniform(
            low=-np.sqrt(6.0 / (input_size + rnn_inner_size * 2)),
            high=np.sqrt(6.0 / (input_size + rnn_inner_size * 2)),
            size=(input_size, rnn_inner_size)), dtype=theano.config.floatX)

        self.W = theano.shared(W_val, 'W', borrow=True)

        U_val = np.asarray(np.random.uniform(
            low=-np.sqrt(6.0 / (input_size + rnn_inner_size * 2)),
            high=np.sqrt(6.0 / (input_size + rnn_inner_size * 2)),
            size=(rnn_inner_size, rnn_inner_size)), dtype=theano.config.floatX)

        self.U = theano.shared(U_val, 'U', borrow=True)

        buw_val = np.zeros((rnn_inner_size,), dtype=theano.config.floatX)
        self.buw = theano.shared(value=buw_val, name='buw', borrow=True)

        Wz_val = np.asarray(np.random.uniform(
            low=-np.sqrt(4.0 / (input_size + rnn_inner_size * 2)),
            high=np.sqrt(4.0 / (input_size + rnn_inner_size * 2)),
            size=(input_size, rnn_inner_size)), dtype=theano.config.floatX)

        self.Wz = theano.shared(Wz_val, 'Wz', borrow=True)

        Uz_val = np.asarray(np.random.uniform(
            low=-np.sqrt(4.0 / (input_size + rnn_inner_size * 2)),
            high=np.sqrt(4.0 / (input_size + rnn_inner_size * 2)),
            size=(rnn_inner_size, rnn_inner_size)), dtype=theano.config.floatX)

        self.Uz = theano.shared(Uz_val, 'Uz', borrow=True)

        bz_val = np.zeros((rnn_inner_size,), dtype=theano.config.floatX)
        self.bz = theano.shared(value=bz_val, name='bz', borrow=True)

        Wr_val = np.asarray(np.random.uniform(
            low=-np.sqrt(4.0 / (input_size + rnn_inner_size * 2)),
            high=np.sqrt(4.0 / (input_size + rnn_inner_size * 2)),
            size=(input_size, rnn_inner_size)), dtype=theano.config.floatX)

        self.Wr = theano.shared(Wr_val, 'Wr', borrow=True)

        Ur_val = np.asarray(np.random.uniform(
            low=-np.sqrt(4.0 / (input_size + rnn_inner_size * 2)),
            high=np.sqrt(4.0 / (input_size + rnn_inner_size * 2)),
            size=(rnn_inner_size, rnn_inner_size)), dtype=theano.config.floatX)

        self.Ur = theano.shared(Ur_val, 'Ur', borrow=True)

        br_val = np.zeros((rnn_inner_size,), dtype=theano.config.floatX)
        self.br = theano.shared(value=br_val, name='br', borrow=True)

        def scan_function(input, inter_output, W, U, Wz, Uz, Wr, Ur, buw, bz, br):
            rj = nnet.sigmoid(T.dot(input, Wr) + T.dot(inter_output, Ur) + br)
            zj = nnet.sigmoid(T.dot(input, Wz) + T.dot(inter_output, Uz) + bz)
            htilde = T.tanh(T.dot(input, W) + rj * T.dot(inter_output, U) + buw)
            inter_output = zj * inter_output + (1 - zj) * htilde

            return inter_output

        outputs_info = T.as_tensor_variable(np.asarray(np.zeros(rnn_inner_size), dtype=theano.config.floatX))
        final_inter_output, updates = theano.scan(scan_function, outputs_info=outputs_info, sequences=[input],
                                                  non_sequences=[self.W, self.U, self.Wz, self.Uz, self.Wr, self.Ur,
                                                                 self.buw, self.bz, self.br])

        self.output = final_inter_output[-1]
        self.param = [self.W, self.U, self.Wz, self.Uz, self.Wr, self.Ur,
                      self.buw, self.bz, self.br]


class FinalLayer(object):
    def __init__(self, input, insize, outsize=4):
        Wf_val = np.asarray(np.random.uniform(
            low=-np.sqrt(6.0 / (insize + outsize)),
            high=np.sqrt(6.0 / (insize + outsize)),
            size=(insize, outsize)), dtype=theano.config.floatX)
        self.Wf = theano.shared(Wf_val, 'Wf', borrow=True)

        bf_val = np.zeros((outsize,), theano.config.floatX)
        self.bf = theano.shared(bf_val, 'bf', borrow=True)

        self.output = T.tanh(T.dot(input, self.Wf) + self.bf)
        self.param = [self.Wf, self.bf]


class Network(object):
    def __init__(self, input, insize, outsize=1):
        embedding_size = 100
        recurrent_layer_size = 300
        self.embeddingLayer = EmbeddingLayer(input, insize, embedding_size)
        self.encoderLayer = EncoderLayer(self.embeddingLayer.output, embedding_size, recurrent_layer_size)

        self.finalLayer = FinalLayer(self.encoderLayer.output, recurrent_layer_size)

        self.output = self.finalLayer.output
        self.param = self.embeddingLayer.param + self.encoderLayer.param + self.finalLayer.param

        self.R2 = 0
        for p in self.param:
            self.R2 += T.sum(p ** 2)


def update_learning_rate(new_rate, network, gradient_param_list, sample, l, cost):
    updates = []
    for param, gparam in zip(network.param, gradient_param_list):
        updates.append((param, param - new_rate * gparam))

    print 'Updated learning rate: %f' % new_rate

    return theano.function(inputs=[sample, l],
                           outputs=cost,
                           updates=updates, allow_input_downcast=True)


def test_network(test_model, train_set, train_l, test_set, test_l, log):
    # test model
    print u'Testing model'
    correct = 0
    for s, lb in zip(train_set, train_l):
        # s = np.reshape(s, (1,) + s.shape)
        net_out = test_model(s)
        guess = np.argmax(net_out)

        if guess == np.argmax(lb):
            correct += 1

    print u'Train set accuracy: %f' % (float(correct) / len(train_set))
    log.write(u'Train set accuracy: %f\n' % (float(correct) / len(train_set)))

    correct = 0
    mistakes = dict()
    index = 0
    for s, lb in zip(test_set, test_l):
        net_out = test_model(s)

        if net_out >= 0:
            guess = 1

        else:
            guess = -1

        if guess == lb:
            correct += 1

        else:
            mistakes[index] = (net_out, lb)
        index += 1

    print mistakes
    print u'Test set accuracy: %f' % (float(correct) / len(test_set))
    log.write(u'Test set accuracy: %f\n' % (float(correct) / len(test_set)))


def run_network(train_set, train_l, test_set, test_l, expname=''):
    log = open(expname + '.txt', 'w')
    plotobj = AccuracyPlot(expname + '.txt', expname)
    log.write(expname + '\n')
    learning_rate = 0.001
    R2_coeff = 5e-4
    log.write('Learning rate: %f\n' % learning_rate)
    log.write('R2coeff: %f\n' % R2_coeff)

    print u'Building model...'

    # Symbolic variables
    sample = T.matrix(u'sample')
    l = T.vector(u'l')

    network = Network(sample, train_set[0].shape[1])

    cost = (network.output - l) ** 2 + R2_coeff * network.R2

    test_model = theano.function(inputs=[sample], outputs=network.output, allow_input_downcast=True)

    # Gradients computations
    gradient_param_list = []
    for w in network.param:
        gradient_param_list.append(T.grad(cost, w))

    updates = []
    for param, gparam in zip(network.param, gradient_param_list):
        updates.append((param, param - learning_rate * gparam))

    train_model = theano.function(inputs=[sample, l],
                                  outputs=cost,
                                  updates=updates, allow_input_downcast=True)

    # Training
    print u'Training...'

    epochs = 100

    # Epochs
    last_error = 0
    iter_cost = 3e10
    i = 0
    while i < epochs:
        last_error = iter_cost
        iter_cost = 0

        sn = 0
        for s, lb in zip(train_set, train_l):
            iter_cost += train_model(s, lb)

            print sn
            sn += 1

            if math.isnan(iter_cost):
                exit(1)

        print u'Epoch %d: cost %f' % (i, iter_cost)
        log.write(u'Epoch %d: cost %f\n' % (i, iter_cost))
        i += 1

        test_network(test_model, train_set, train_l, test_set, test_l, log)  # Test after every epoch
        plotobj.update(1)

        if last_error < iter_cost:
            learning_rate /= 2
            train_model = update_learning_rate(learning_rate, network, gradient_param_list, sample, l, cost)

    log.close()



