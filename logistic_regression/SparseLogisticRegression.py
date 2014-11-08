'''
This is a logistic regression that takes a sparsely encoded matrix where
each value in the row correlates to the presense of a value 1.

i.e.

x = [1,56,291,111856] represents a vector of at least size 111857 with four 1's 
located at index 1,56,291 and 111856.

This logistic regression was designed to work well with vowpal wabbit's hash trick:

https://github.com/JohnLangford/vowpal_wabbit/

and using an adaptive learning rate mentioned in this google paper:

http://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf

This code is loosely based on code I took from: 

https://www.kaggle.com/users/185835/tinrtgu

My contributions are as follows: de-couping the code from the Critero Competition,
making it look like sklearn, adding l1/l2 regularization and minibatching


Dependencies: 

Numpy

'''
import numpy
from math import exp, log, sqrt, ceil
from datetime import datetime

class SparseLogisticRegression(object):
    
    #Verbose mode determines the logloss at every step and occasionally prints it out. 
    #Setting verbose = False will give performance gains
    def __init__(self, dimension_size, learning_rate=.1, max_epochs=1, l1_ratio=0., l2_ratio=.0, verbose=True):
        self.D = dimension_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.l1_ratio = l1_ratio
        self.l2_ratio = l2_ratio
        self.verbose = verbose
        
        self.w = [0.] * self.D  # weights
        self.n = [0.] * self.D  # number of times we've encountered a feature
        
        return 
    
    def logloss(self,p, y):
        p = max(min(p, 1. - 10e-12), 10e-12)
        return -log(p) if y == 1. else -log(1. - p)
    
    def update_w(self,x, p, y):
        for i in x:
            # alpha / (sqrt(n) + 1) is the adaptive learning rate heuristic
            # (p - y) * x[i] is the current gradient
            # note that in our case, if i in x then x[i] = 1
            self.w[i] -= (p - y) * self.learning_rate / (sqrt(self.n[i]) + 1.)
            self.n[i] += 1.
    
    def get_p(self,x):
        wTx = 0.
        for i in x:  # do wTx
            wTx += self.w[i] * 1.  # w[i] * x[i], but if i in x we got x[i] = 1.
        return 1. / (1. + exp(-max(min(wTx, 20.), -20.)))  # bounded sigmoid
    
    def fit(self,X, Y):
        if X != numpy.ndarray:
            X = numpy.array(X)
        rows = X.shape[0]
        verbose_freq = ceil(rows/10)
        for current_epoch in range(self.max_epochs):
            loss = .0
            if self.verbose == True:
                print('%s\tepoch: %s' % (datetime.now(), current_epoch))
            for index in range(rows):
                p = self.get_p(X[index])
                self.update_w(X[index, :], p, Y[index])
                
                if self.verbose == True:
                    loss += self.logloss(p,Y[index])
                    if index % verbose_freq == 0 and index > 0:
                        print('%s\tencountered: %d\tcurrent logloss: %f' % (datetime.now(), index, loss/index))

        
    def predict_proba(self, X):
        if X != numpy.ndarray:
            X = numpy.array(X)
        rows = X.shape[0]
        preds = []
        for index in range(rows):
            preds.append(self.get_p(X[index,:]))
            
        return preds
    
    def predict(self, X):
        preds = self.predict_proba(X)
        classes = []
        for pred in preds:
            if pred >= .5:
                classes.append(1)
            else:
                classes.append(0)
                
        return classes
    