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
        self.sentenceLength = T.shape(final_inter_output)[0]
        self.param = [self.W, self.U, self.Wz, self.Uz, self.Wr, self.Ur,
                      self.buw, self.bz, self.br]


class DecoderBlock(object):
    def __init__(self, input, recurrent_layer_size, sentence_length):
        c = input
        input_size = recurrent_layer_size * 2 + 1  # h_t-1, y_t-1, c
        output_size = 1

        Woutput_val = np.asarray(np.random.uniform(
            low=-np.sqrt(6.0 / (input_size + output_size)),
            high=np.sqrt(6.0 / (input_size + output_size)),
            size=(input_size, output_size)), dtype=theano.config.floatX)

        self.Woutput = theano.shared(Woutput_val, 'Woutput', borrow=True)

        boutput_val = np.zeros((output_size,), dtype=theano.config.floatX)
        self.boutput = theano.shared(value=boutput_val, name='boutput', borrow=True)

        Witer_val = np.asarray(np.random.uniform(
            low=-np.sqrt(6.0 / (input_size + recurrent_layer_size)),
            high=np.sqrt(6.0 / (input_size + recurrent_layer_size)),
            size=(input_size, recurrent_layer_size)), dtype=theano.config.floatX)

        self.Witer = theano.shared(Witer_val, 'Witer', borrow=True)

        biter_val = np.zeros((recurrent_layer_size,), dtype=theano.config.floatX)
        self.biter = theano.shared(value=biter_val, name='biter', borrow=True)

        def scan_function(h_t, y_t, c, Woutput, boutput, Witer, biter):
            total_input = T.concatenate((c, h_t, y_t.reshape((1,))))
            h_t1 = T.tanh(T.dot(total_input, Witer) + biter)
            y_t1 = T.tanh(T.dot(total_input, Woutput) + boutput)[0]

            return [h_t1, y_t1]

        outputs_info = [T.as_tensor_variable(np.asarray(np.zeros(recurrent_layer_size), dtype=theano.config.floatX),
                                             name='ht', ndim=1),
                        T.as_tensor_variable(np.asarray(0.0, dtype=theano.config.floatX))]
        final_inter_output, updates = theano.scan(scan_function, outputs_info=outputs_info,
                                                  non_sequences=[input, self.Woutput, self.boutput, self.Witer,
                                                                 self.biter],
                                                  n_steps=sentence_length)

        # final_inter_output[0] = final_inter_output[0].flatten()
        # final_inter_output[1] = final_inter_output[1].flatten()

        self.output = final_inter_output[1]
        self.param = [self.Woutput, self.boutput, self.Witer, self.biter]


class Network(object):
    def __init__(self, input, insize):
        embedding_size = 100
        recurrent_layer_size = 300
        self.embeddingLayer = EmbeddingLayer(input, insize, embedding_size)
        self.encoderLayer = EncoderLayer(self.embeddingLayer.output, embedding_size, recurrent_layer_size)

        self.decoderBlock = DecoderBlock(self.encoderLayer.output, recurrent_layer_size,
                                         self.encoderLayer.sentenceLength)

        self.output = self.decoderBlock.output
        self.param = self.embeddingLayer.param + self.encoderLayer.param + self.decoderBlock.param

        self.R2 = 0
        for p in self.param:
            self.R2 += T.sum(p ** 2)


def update_learning_rate(new_rate, network, gradient_param_list, sample, l, cost):
    updates = []
    for param, gparam in zip(network.param, gradient_param_list):
        updates.append((param, param - new_rate * gparam))

    print 'Updated learning rate: %f' % new_rate

    return theano.function(inputs=[sample, l], outputs=cost, updates=updates, allow_input_downcast=True)


def test_network(test_model, train_set, test_set, log):
    # test model
    print u'Testing model'

    correct = 0
    mistakes = dict()
    index = 0
    for s in test_set:
        net_out = test_model(s.getInputArray())
        golden_label = s.getLabelsArray()

        net_out[net_out >= 0.0] = 1
        net_out[net_out < 0.0] = -1

        if np.sum(net_out == golden_label) == len(net_out):
            correct += 1

        else:
            mistakes[index] = (net_out, golden_label)
        index += 1

    print mistakes
    print u'Test set accuracy: %f' % (float(correct) / len(test_set))
    log.write(u'Test set accuracy: %f\n' % (float(correct) / len(test_set)))


def run_network(train_set, test_set, expname):
    # theano.config.mode = 'DebugMode'

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

    network = Network(sample, train_set[0].getTotalInputLength())

    cost = T.sum((network.output - l) ** 2)

    test_model = theano.function(inputs=[sample], outputs=network.output, allow_input_downcast=True)

    test_network(test_model, train_set, test_set, log)

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

    epochs = 2

    # Epochs
    last_error = 0
    iter_cost = 3e10
    i = 0
    while i < epochs:
        last_error = iter_cost
        iter_cost = 0

        sn = 0
        for s in train_set:

            iter_cost += train_model(s.getInputArray(), s.getLabelsArray())

            print sn
            sn += 1

            if math.isnan(iter_cost):
                exit(1)

        print u'Epoch %d: cost %f' % (i, iter_cost)
        log.write(u'Epoch %d: cost %f\n' % (i, iter_cost))
        i += 1

        test_network(test_model, train_set, test_set, log)  # Test after every epoch
        plotobj.update(1)

        if last_error < iter_cost:
            learning_rate /= 2
            train_model = update_learning_rate(learning_rate, network, gradient_param_list, sample, l, cost)

    log.close()



