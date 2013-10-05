from __future__ import division         # for float divisions
'''
Created on Mar 23, 2013

@author: Jason
'''
from myExceptions import *
import numpy as np
import pickle as pkl
from numpy.linalg import norm as l2norm


class ANN(object):
    '''
    The ANN class represents an approximate nearest neighbor classifier. The training phase
    consists of simply storing the data. At testing time, given a distance R and a query point q, 
    the classifier finds at most L R-nearest neighbors and reports on the query point's class
    by voting over these neighbors. In the case of a tie, we flip a coin.
    
    The hyper-parameter of this classifier is the number of approximate nearest neighbors L
    to consider during voting.
    
    @author:  Jason Filippou
    @contact:  jasonfil@cs.umd.edu
    @since March 23, 2013
    The space is assumed Euclidean and all distances are measured by the l2 norm.
    @ivar L: The number of approximate nearest neighbors to use for classification (tuneable parameter)
    @ivar R: The radius used to find ANNs (non-tuneable parameter)
    @ivar numAlonePoints: the number of points that had no R-near neighbors (useful primarily for statistical analysis)
    @ivar trainDat: A 2D numpy array which holds the classifier's training data. Patterns in rows, features in columns.
    @ivar trainLabs: a 1D numpy array, in row correspondence with trainDat, which holds the labels of the training data.
    '''

    def __init__(self, R, L = None):
        '''
        Constructor of the object
        @raise: LogicalError if R is 0 or None
        '''
        
        if R is None or R == 0:
            raise LogicalError, "ANN constructor: We need a positive distance threshold R."
        self.L = L # L = None means that we will get all R-nearest neighbors. 
        self.R = R
        self.numAlonePoints = 0
        self.printAloneMsg = True
        
    def __vote__(self, pointIndices):
        
        '''
        The points provided get to vote on a binary label (+1, -1) which is then returned.
        If there are no points, this indicates that the point had no R-near neighbors,
        which we deal with by randomly voting.
        '''
        
        if len(pointIndices) == 0:
            self.numAlonePoints = self.numAlonePoints + 1
            return 2*np.random.randint(0, 2) - 1
        labels = self.trainLabs[pointIndices]
        positiveVotes = len(labels > 0)
        negativeVotes = len(labels < 0)
        if positiveVotes == negativeVotes:
            return 2*np.random.randint(0, 2) - 1      # flip a coin which will land on either +1 or -1
        else:
            return np.maximum(positiveVotes, negativeVotes)
            
        
        
    def train(self, trainDat, trainLabs):
        '''
        Nearest neighbor classifiers have the advantage that they have virtually
        no training phase, other than storing the training data.
        @param trainDat 2D numpy array containing the training data
        @param trainLabs 1D numpy array containing the training data labels
        @return the model itself
        '''
        
        self.trainDat = trainDat
        self.trainLabs = trainLabs
        return self
    
    def tune(self, validationData, validationLabels, hyperparamsToTest):
        '''
        Given validation data and labels, test all hyperparameters provided
        on the data and maintain the hyperparameter that provided for the best
        results.
        @param validationData a 2D numpy array containing the validation data
        @param validationLabels a 1D numpy array containing the validation data labels
        @param hyperparamsToTest an iterable containing the entire array of hyper-parameters 
                (values for L) to test.
        @raise LogicalError if hyperparamsToTest is not an iterable, None or Empty
        @return the optimal value of L
        '''
        
        try:
            len(hyperparamsToTest)
        except TypeError:
            raise LogicalError, "Method ANN.tune(): need to provide an iterable for third argument."
        self.printAloneMsg = False
        if hyperparamsToTest is None or len(hyperparamsToTest) == 0:
            raise LogicalError, "Method ANN.tune(): need to provide hyper-parameters."
        bestAccuracy = 0.0
        for h in hyperparamsToTest:
            accuracy = self.test(validationData, validationLabels, h)
            if accuracy > bestAccuracy:
                bestAccuracy = accuracy
                self.L = h
        self.printAloneMsg = True
        return self.L
    
    def test(self, testData, testLabels, L):
        '''
        Testing on a nearest neighbor classifier involves finding L R-nearest neighbors 
        of every query point. Alternatively, all R-nearest neighbors could be found.
        @param testData a 2D ndarray representing the testing data
        @param testLabels a 1D ndarray representing the testing labels: one cell per testData row
        @param L the number of approximate nearest neighbors (optional)
        @raise LogicalError if there are not exactly as many labels as there are examples
        @return the accuracy reported on the testData
        '''
        
        if len(testData) != len(testLabels):
            raise LogicalError, "Method ANN.test(): There needs to be one label per testing example"
        if L is None:
            L = self.L          # Use stored hyper-parameter if not provided with one.
                                # Note that L could still be None, in which case we do not
                                # want to stop after finding L approximate nearest neighbors.
        appNeighbors = dict()
        correctPreds = 0
        for i in range(len(testData)): 
            appNeighbors[i] = list()
            for j in range(len(self.trainDat)):
                if l2norm(testData[i]-self.trainDat[j]) <= self.R:      
                    appNeighbors[i].append(j)
                    if L is not None: L = L - 1 # Found an approximate neighbor
                    if L == 0: break # This is exactly where the computational advantage of having at most L approx. nearest neighbors is introduced
            votedLabel = self.__vote__(appNeighbors[i])
            if votedLabel == testLabels[i]:
                correctPreds = correctPreds + 1
        
        if self.printAloneMsg == True:    
            print "For radius = " + str(self.R) + ", points found alone: " + str(self.numAlonePoints)
        self.numAlonePoints = 0                 # reset counter
        return correctPreds / len(testData)           
                    
       
if __name__ == '__main__':
    
    np.random.seed(47) # to make experiments reproducible
    
    ### Step 1: Read data ####
    
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
    
    ### Step 3:  Initialize and tune the ANN classifier ###
     
    classifier = ANN(20600)
    classifier.train(trainData, trainLabels)
    optimalL = classifier.tune(validData_new, validLabels_new)
    print "The optimal value of L found for R=20600 was: " + str(optimalL)
    accuracy = classifier.test(testData, testLabels, optimalL)
    print "Accuracy found for this optimal L: " + str(accuracy)
