from __future__ import division         # for float divisions

'''
Created on Mar 26, 2013

@author: Jason
'''

import numpy as np
import pickle as pkl
from myExceptions import *

class LSH_ANN(object):
    
    '''
    An approximate nearest neighbors classifier which works with locality-sensitive hashing.
    
    @author: Jason Filippou
    @contact: jasonfil@cs.umd.edu
    @since: March 23, 2013
    @ivar L: The number of hash tables in the data structure
    @ivar k: The number of dot products per hash table (essentially, the dimensionality of each
            hash table's keys)
    @ivar randomVectorMap: A dictionary that maps every hash table to k randomly generated D-dimensional
            vectors which are used to produce k random projections of the training points. We need to maintain this information
            after training because we need to take k inner products of the testing point with the exact same k random vectors.
    @ivar randomBiasMap: Similar to randomVectorMap in that it stores the b parameter, so that we can quantize the test point
            projection in exactly the same way.
    @ivar w: The width of projection (tuneable parameter)
    @ivar trainLabs: a 1D numpy array,  holding one label per pattern.
    @ivar trainDat: the training data in the format required by LSH (array of dictionaries)
    '''


    def __init__(self, L, k, w):
        
        '''
        Initializes the data structure parameters.
        We follow a lazy approach in that we do not initialize the data structure itself until
        training time.
        
        @param L the number of hash tables
        @param k the number of dot products to generate per hash table
        @param w the width of projection to use at training time.
        @return the object itself
        '''
        
        self.L = L
        self.k = k
        self.w = w
        self.numAlonePoints = 0
        
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
         Trains the LSH classifier. This involves randomly generating L*K D-dimensional vectors whose components 
         are drawn i.i.d from a Gaussian and initializing the hash tables. Every hash table will be associated with
         a different set of K such vectors, yielding K different inner products (random proections on the real number line).
         This is arguably the most complex method of this class.
         
         @param trainDat: a 2D numpy array holding our training data. Patterns in rows, features in columns.
         @param trainLabs: a 1D numpy array, in row correspondence with trainDat, holding one label per pattern.
         @raise LogicalError if the training labels are not in row correspondence with training data.
        '''
        
        if len(trainDat) != len(trainLabs):
            raise LogicalError, "Method LSH_ANN.train(): The training labels were not in row correspondence with the training data."
         
        self.trainLabs = trainLabs
        self.trainDat = list()     
        self.randomVectorMap = dict()
        self.randomBiasMap = dict()
        
        # For every hash table, generate k random D-dimensional vectors, whose components are drawn i.i.d from a Gaussian distribution.
        # We associate each of these vectors with the relevant hash table by storing the association in the "randomVectorMap" dictionary.
        # TODO: In order to make the experiments re-producible, the random number seed needs to be initialized to a static number.
        
        for hashInd in range(self.L):
            b = np.random.uniform(0, self.w)
            randomVecs = np.array([np.random.normal(0, 1, len(trainDat[0])) for _ in range(self.k)]) # k random D-dimensional vectors
            self.randomVectorMap[hashInd] = randomVecs
            self.randomBiasMap[hashInd] = b
            
            # Multiplying the k X D "randomVecs" matrix by our training data transposed (to make it D x N)
            # yields a k X N matrix which describes every point by k random projections onto R1.
            # We then transpose the result to make it N x k.
            projMat = np.array((randomVecs.dot(trainDat.T))).T
            
            # We now quantify every projection into buckets according to  the formula (u.v + b)/w
            projMat = np.array([np.floor((row + b)/self.w) for row in projMat])
            
            # Now, every example can be stored in the "hashInd"'th hash table s.t 
            # its k-dimensional projection is the key and its index is the mapped value
            
            currentDict = dict()
            for exIndex in range(len(projMat)):
                if tuple(projMat[exIndex]) not in currentDict:
                    currentDict[tuple(projMat[exIndex])] = list()
                currentDict[tuple(projMat[exIndex])].append(exIndex)
            self.trainDat.append(currentDict) # Store the dictionary
        
        
    def test(self, testDat, testLabs, neighbNum = None):
        
        '''
        Given testing examples and labels, classify every example and report on classification accuracy.
        In LSH_ANN, testing occurs by looping through all the hash tables, producing k inner products of the
        test point and checking to see whether there exist collisions. Collisions indicate R-near neighbors, 
        with probability dependent on the choice of L and k.
        @param testDat: a 2D numpy array which holds the testing data. Patterns in rows, variables in columns.
        @param testLabs: a 1D numpy array, in row correspondence with testDat, that contains the labels of the patterns.
        @param neighbNum (optional): The maximum number of approximate nearest neighbors to consider in the label voting of the example.
                if None (the default), then all neighbors found are considered.
        @return: the accuracy of the classification.
        @raise LogicalError: If the testing labels are not in row correspondence with the testing data.
        @note: The Jan 2008 Communications of the ACM paper of Andoni and Indyk imply that we can also consider votes from duplicates. That is,
        if we find the same R-near neighbor more than once, we should consider his vote as many times as he appears.
        '''
        
        if len(testLabs) != len(testDat):
            raise LogicalError, "Method LSH_ANN.test(): The testing labels are not in row correspondence with the testing examples."
        
        # We need to loop through all hash tables, find the k different quantized projections of the testing point and decide
        # on whether the exact same projection exists within a given hash table. If so, this collision indicates, with high probability,
        # the existence of an approximate nearest neighbor for the testing point. In this case, we store the training point's index
        # and eventually use them all for voting.
        
        correctPreds = 0
        self.randomVotes = 0
        for i in range(len(testDat)):
            neighbors = list()
            for htIndex in range(self.L):
                testPointProjections = np.array([testDat[i].dot(v) for v in self.randomVectorMap[htIndex]])  
                testPointProjections = np.floor((testPointProjections + self.randomBiasMap[htIndex]) / self.w) # TODO: Make into one line after debugging. 
                if tuple(testPointProjections) in self.trainDat[htIndex]:
                    assert len(self.trainDat[htIndex][tuple(testPointProjections)]) > 0
                    if neighbNum is not None: # add all neighbors up to a point
                        neighbors = neighbors + self.trainDat[htIndex][tuple(testPointProjections)][0:neighbNum]
                        neighbNum = neighbNum - len(self.trainDat[htIndex][tuple(testPointProjections)])
                        if neighbNum <= 0:  break       # Finished scanning neighbors for test point
                    else: # add all neighbors
                        neighbors = neighbors + self.trainDat[htIndex][tuple(testPointProjections)]        
            pointLabel = self.__vote__(neighbors)
            if pointLabel == testLabs[i]:
                correctPreds = correctPreds + 1
         
        print "Randomly voted " + str(self.randomVotes) + " times."
        return correctPreds / len(testDat)               
        
    def tune(self, validDat, validLabs, hyperParamsToTest):
        
        '''
        Tune the LSH classifier for the optimal value of w (the quantization bin width). 
        Sets self.w to the value that minimized validation error.
        @param validDat a 2D numpy array holding validation data. Patterns in rows, variables in columns.
        @param validLabs a 1D numpy array, in row correspondence with validDat, which holds validation data labels.
        @param hyperParamsToTest: an iterable containing all the different values of w to tune on.
        @raise LogicalError: If the labels are not in row correspondence with the validation data
        '''
        
        if len(validLabs) != len(validDat):
            raise LogicalError, "Method LSH_ANN.tune(): Labels were not in row correspondence with data."
        
        bestAccuracy = 0
        bestW = self.w
        for val in hyperParamsToTest:
            self.w = val
            accuracy = self.test(validDat, validLabs)   # Not providing neighbor number for now.
            if accuracy > bestAccuracy:
                bestAccuracy = accuracy
                bestW = val
        self.w = bestW
        return bestW

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
    
    ### Tune the classifier for w, using all R-near neighbors.
    
#     for w in range(4, 10):
#         classifier = LSH_ANN(80, 15, w)
#         classifier.train(trainData, trainLabels)
#         accuracy = classifier.test(validData_new, validLabels_new)
#         print "Attained an accuracy of: " + str(accuracy) + " for a bucket width of " + str(w)

    ### Tune the classifier for L' (number of R-near neighbors) while keeping a bin width of 5
    
    for n in range(1, 11, 1):
        classifier = LSH_ANN(80, 15, 9)
        classifier.train(trainData, trainLabels)
        accuracy = classifier.test(validData_new, validLabels_new)
        print "Attained an accuracy of: " + str(accuracy) + " for a number of neighbors = " + str(n) + " and a bucket width of 5." 