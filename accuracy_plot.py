__author__ = 'brtdra'

import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as pypl
import re


class AccuracyPlot(object):
    def __init__(self, logfile, expname):
        self.filename = logfile
        self.expname = expname
        pypl.ion()

    def plotTestAccuracy(self, epoch_interval=2):
        inf = open(self.filename)

        accuracies = []

        for l in inf:
            mtc = re.match(r'Test set accuracy: (.+)', l)

            if mtc is not None:
                accuracies.append(float(mtc.group(1)))

        inf.close()

        y = np.array(accuracies)
        pypl.figure(0)
        pypl.clf()
        pypl.plot(np.arange(0, epoch_interval * len(y), epoch_interval), y)
        pypl.title(self.expname + 'test set accuracy')
        pypl.xlabel('epoch')
        pypl.ylabel('test set accuracy')
        pypl.ylim(ymax=1.0)

        # pypl.show()
        pypl.savefig(self.expname + '_test_accuracy.eps')

    def plotError(self):
        inf = open(self.filename)

        errors = []

        for l in inf:
            mtc = re.match(r'Epoch (\d+): cost (.+)', l)

            if mtc is not None:
                errors.append(float(mtc.group(2)))

        inf.close()

        y = np.array(errors)
        pypl.figure(1)
        pypl.clf()
        pypl.plot(np.arange(0, len(y)), y)
        pypl.title(self.expname + 'error function')
        pypl.xlabel('epoch')
        pypl.ylabel('error value')
        pypl.ylim(ymin=0.0)

        # pypl.show()
        pypl.savefig(self.expname + '_error_value.eps')

    def update(self, epoch_interval=2):
        self.plotTestAccuracy(epoch_interval)
        self.plotError()

