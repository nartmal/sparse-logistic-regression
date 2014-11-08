#!/usr/bin/python
import numpy
from datetime import datetime
from math import exp, log, sqrt
from sklearn import cross_validation
import hashlib
import scipy as sp
from logistic_regression.SparseLogisticRegression import SparseLogisticRegression

fieldNames = [
        "id",#47686351 / 1.0                    /0  / 0
        "click",#2 / 10^-7                      /1 
        "hour",#249 / 10^-6                     /2  / 1
        "C1",#63 / 10^-6                        /3  / 2
        "banner_pos",#63 / 10^-6                /4  / 3
        "site_id",#29953 / .000628              /5  / 4
        "site_domain",#36764 / .0007707         /6  / 5
        "site_category",#217 / 10^-6            /7  / 6
        "app_id",#32147 / .00067413             /8  / 7 
        "app_domain",#1927 / 10^-5              /9  / 8 
        "app_category",#281 / 10^-6             /10 / 9
        "device_id",#3201440 / .067             /11 / 10
        "device_ip",#8521254 / .1786            /12 / 11
        "device_os",#105 / 10^-6                /13 / 12
        "device_make",#2468 / 10^-5             /14 / 13
        "device_model",#57138 / .001198         /15 / 14
        "device_type",#40 / 10^-7               /16 / 15
        "device_conn_type",#40 / 10^-7          /17 / 16
        "device_geo_country",#2267 / 10^-5      /18 / 17
        "C17",#10967 / .0002299                 /19 / 18
        "C18",#68 / 10^-6                       /20 / 19
        "C19",#78 / 10^-6                       /21 / 20
        "C20",#2214 / 10^-5                     /22 / 21
        "C21",#40 / 10^-7                       /23 / 22
        "C22",#367 / 10^-6                      /24 / 23
        "C23",#1636 / 10^-5                     /25 / 24
        "C24"#288 / 10^-6                       /26 / 25
        ]

FILENAME = "/home/lam/Desktop/Kaggle/avazu_ctr_prediction/train_sample_tiny.csv"
TEST_FILE = "/home/lam/Desktop/Kaggle/avazu_ctr_prediction/test_rev2.csv"
SUBMISSION_FILE = "/home/lam/Desktop/Kaggle/avazu_ctr_prediction/submission.csv"

useColsTuple = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26)
testUseColsTuple = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25)


def get_x(row, D):
    x = [0]  # 0 is the index of the bias term
    counter = 2
    for value in row:
        
        if counter == 2:
            value = value[6:]
        
        index = int(hashlib.sha224(value + fieldNames[counter]).hexdigest(), 16) % D  # weakest hash ever ;)
        x.append(index)
        counter += 1
    
    #quadratics
    starting_index = 17
    
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+1] + "C17xC18").hexdigest(), 16) % D)
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+2] + "C17xC19").hexdigest(), 16) % D)
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+3] + "C17xC20").hexdigest(), 16) % D)
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+4] + "C17xC21").hexdigest(), 16) % D)
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+5] + "C17xC22").hexdigest(), 16) % D)
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+6] + "C17xC23").hexdigest(), 16) % D)
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+7] + "C17xC24").hexdigest(), 16) % D)
    
    starting_index = 18
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+1] + "C18xC19").hexdigest(), 16) % D)
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+2] + "C18xC20").hexdigest(), 16) % D)
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+3] + "C18xC21").hexdigest(), 16) % D)
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+4] + "C18xC22").hexdigest(), 16) % D)
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+5] + "C18xC23").hexdigest(), 16) % D)
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+6] + "C18xC24").hexdigest(), 16) % D)
    
    starting_index = 19
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+1] + "C19xC20").hexdigest(), 16) % D)
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+2] + "C19xC21").hexdigest(), 16) % D)
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+3] + "C19xC22").hexdigest(), 16) % D)
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+4] + "C19xC23").hexdigest(), 16) % D)
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+5] + "C19xC24").hexdigest(), 16) % D)
    
    starting_index = 20
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+1] + "C20xC21").hexdigest(), 16) % D)
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+2] + "C20xC22").hexdigest(), 16) % D)
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+3] + "C20xC23").hexdigest(), 16) % D)
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+4] + "C20xC24").hexdigest(), 16) % D)
    
    starting_index = 21
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+1] + "C21xC22").hexdigest(), 16) % D)
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+2] + "C21xC23").hexdigest(), 16) % D)
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+3] + "C21xC24").hexdigest(), 16) % D)

    starting_index = 22
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+1] + "C22xC23").hexdigest(), 16) % D)
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+2] + "C22xC24").hexdigest(), 16) % D)

    starting_index = 23
    #x.append(int(hashlib.sha224(row[starting_index] + row[starting_index+1] + "C23xC24").hexdigest(), 16) % D)
    
    return x  # x contains indices of features that have a value of 1
    
def generate_submission(D, model):
    print "\n%s\topening test csv %s..." % (datetime.now(),TEST_FILE)
    data = numpy.genfromtxt(TEST_FILE, delimiter=',',dtype=str , usecols=testUseColsTuple)
    print "%s\tconverting to numpy array"%(datetime.now())
    dataset = numpy.array(data)
    
    print "%s\tencoding..."%(datetime.now())
    rows, cols = dataset.shape
    
    X = []
    ids = []
    for index in range(rows):
        ids.append(dataset[index, 0])
        X.append(get_x(dataset[index, 1:], D))
    
    X = numpy.array(X,copy=False)
    
    rows,cols = X.shape
    print "%s\tpredicting..."%(datetime.now())

    prob = model.predict_proba(X)
    
    print "%s\twriting submission file..."%(datetime.now())
    with open(SUBMISSION_FILE, "w") as submission:
        submission.write('id,click\n')
        for i in range(len(ids)):
            submission.write( "%s,%s\n"%(ids[i],prob[i]) )
    
    return 0
    

#FROM KAGGLE
def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll

def train(D):
    
    print "%s\topening csv %s..." % (datetime.now(),FILENAME)
    
    data = numpy.genfromtxt(FILENAME, delimiter=',',dtype=str, usecols=useColsTuple )
    print "%s\tconverting to numpy array"%(datetime.now())
    dataset = numpy.array(data)
    
    print "%s\tencoding..."%(datetime.now())
    rows, cols = dataset.shape
    
    X = []
    Y = []
    for index in range(rows):

        Y.append(1. if dataset[index,0] == '1' else 0.)
        X.append(get_x(dataset[index,1:], D))
    
    
    print "%s\tstarting logistic regression..."%(datetime.now())
    #start log regression
    allTestLogLoss = []
    for i in range(3):
        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, Y, test_size=.20)

        model = SparseLogisticRegression(D)
        model.fit(X_train,y_train)
        preds = model.predict_proba(X_cv)
        logloss = llfun(y_cv, preds)
        
        allTestLogLoss.append(logloss)
        print "%s\ttesting logloss for iter %s:\t%s"%(datetime.now(),i + 1,logloss)

    print "\n%s\taverage/std training logloss:\t %s / %s" % (datetime.now(),numpy.mean(allTestLogLoss), numpy.std(allTestLogLoss))
    
    return model

def main():
    D = 2 ** 24
    model = train(D)
    generate_submission(D,model)##REMEBER TO CHANGE RANGE
    
    return 0
    
if __name__ == "__main__":
    main()
    
