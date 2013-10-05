'''
Created on Mar 23, 2013

@author: Jason
'''

import numpy as np 
from ANN import ANN
from myExceptions import *
import pickle as pkl
from numpy.linalg import norm as l2norm
from scipy.spatial.distance import pdist

def train(trainDat, trainLabs, R, hyperparam):
    '''
    Train a classifier of your choice on the data.
    @param trainDat an ndarray of training data, one row per example
    @param trainLabs an ndarray of training data labels, one row per training example
    @param R the input to the ANN classification algorithm (radius)
    @param hyperparam the hyper-parameter to train with (number of neighbors to use)
    @raise LogicalError if the training labels do not match the training data
    @return the trained classifier 
    '''
    
    if len(trainLabs) != len(trainDat):
        raise LogicalError, "Method \"train\": training labels don't match with training data."
    classifier = ANN(R, hyperparam) # TODO: Solve the radius situation
    classifier = classifier.train(trainDat, trainLabs)
    return classifier
    
def test(classifier, testData, testLabs):
    '''
    Test a classifier on the data provided.
    @param classifier a classifier object
    @param testData a 2D ndarray holding the data to test on
    @param testLabs a 1D ndarray holding the labels of the data that we test on
    @raise LogicalError if the test data labels do not numerically match the testing data
    @return the accuracy reported
    '''
    
    if len(testData) != len(testLabs):
        raise LogicalError, "Method test(): test data labels do not match examples."
    return classifier.test(testData, testLabs, None)
    
    

def tune(classifier, validationData, validationDataLabels, hyperParamList):
    
    '''
    Tune a trained classifier on validationData by choosing the hyperparameter which yielded
    the best results.
    @param classifier the trained classifier
    @param validationData the data to validate on
    @param validationDataLabels the validation data's labels
    @param hyperParamList the list of hyperparameters to choose from
    @raise LogicalError if there exists no 1-1 mapping between the validation data and its labels
    '''
    
    if len(validationData) != len(validationDataLabels):
        raise LogicalError, "Method tune(): validationDataLabels do not match validation examples"
    
    return classifier.tune(validationData, validationDataLabels, hyperParamList)

def cross_validate(data, k, hyperparams, labels):
    '''
    Take a bunch of data, run k-fold cross validation on it
    and then return the maximum accuracy score. 
    @param data 2D numpy array holding the full array of data
    @param k the number of folds
    @param hyperparams a list containing the different hyper-parameters to test
    @param labels an ndarray of data labels, one per row of the "data" ndarray
    @return a tuple (bestHyperParam, bestAcc) representing the optimal amount of neighbors as well as 
                        the maximum accuracy scored over all the folds
    '''
    bestAccuracy = 0.0
    optimalHyperParam = None
    for hyperparam in hyperparams: # choose optimal hyper-parameter
        accuracies = list() 
        for fold in range(k): # fold = [0, 1,...,k-1]
            trainDat = [data[i] for i in range(len(data)) if i%k != fold] 
            trainLabs = [labels[i] for i in range(len(data)) if i%k != fold]
            testDat = [data[i] for i in range(len(data)) if i%k == fold] # test every k-th example
            testLabs = [labels[i] for i in range(len(data)) if i%k == fold] 
            model = train(trainDat, trainLabs, hyperparam)
            accuracies.append(test(model, testDat, testLabs))
        accuracy = np.mean(accuracies)
        if accuracy > bestAccuracy:         # store the data
            bestAccuracy = accuracy
            optimalHyperParam = hyperparam
    
    return (optimalHyperParam, bestAccuracy )


if __name__ == '__main__':
    
    try: 
        
        np.random.seed(47) # To make results reproducible
        ### Step 1: Read the stored data ####
        
        trainData = pkl.load(open('proc_data/trainDat.pkl'))
        trainLabels = pkl.load(open('proc_data/trainLabs.pkl'))
        validData = pkl.load(open('proc_data/validDat.pkl'))
        validLabels = pkl.load(open('proc_data/validLabs.pkl'))
        print "Data loaded!"
        
        ### Step 2: Split the validation data in two, creating actual 
        ### validation data on which to validate and test data on which to test 
        
        indices = [index for (index, _) in list(enumerate(validLabels))]
        indices = np.random.permutation(indices)
        validIndices = indices[0:500]
        testIndices = indices[500:1000]
        validData_new = validData[validIndices]
        validLabels_new = validLabels[validIndices]
        testData = validData[testIndices]
        testLabels = validLabels[testIndices]
        print "Split performed!"
        print "#Validation examples: " + str(len(validData_new))
        print "#Testing examples: " + str(len(testData))
        
        ### Step 3: Tune on the validation data and test on the training data, reporting accuracy ###
        
        for R in range(20000, 22000, 100):  # Test for a number of radii
            ANNClassifier = train(trainData, trainLabels,R, None)
            print "Trained ANN classifier for R = " + str(R)
            optimalL = tune(ANNClassifier, validData_new, validLabels_new, range(1, 11))        # test all L from 1 to 10
            print "The optimal value of L found for R = " + str(R) + " was: " + str(optimalL)
            #pkl.dump(ANNClassifier, open('proc_data/ANNclassifier.pkl', 'wb'))
            #print "Stored ANN classifier on disk."
            #ANNClassifier = pkl.load(open('proc_data/ANNclassifier.pkl', 'rb'))
            accuracy = test(ANNClassifier, testData, testLabels)
            print "For R = " + str(R) + " we report an accuracy of: " + str(accuracy) + " on the testing data."
        print "All done!"
    except Exception as e:
        print "An exception occurred: " + str(e)
    
    