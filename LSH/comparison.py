'''
Created on Apr 4, 2013

@author: Jason
'''
import pickle as pkl
import time as t
import numpy as np
from ANN import ANN
from LSH_ANN import LSH_ANN

if __name__ == "__main__":
    
    try:
        np.random.seed(47) # To make results reproducible
            
        # Step 1: Read the stored data 
            
        trainData = pkl.load(open('proc_data/trainDat.pkl'))
        trainLabels = pkl.load(open('proc_data/trainLabs.pkl'))
        validData = pkl.load(open('proc_data/validDat.pkl'))
        validLabels = pkl.load(open('proc_data/validLabs.pkl'))
        print "Data loaded!"
        
        # Step 2: Split the validation data into actual validation and
        # testing data
        
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
        
        # Step 3: train both classifiers on the entire array of data
        
        concatData = np.concatenate([trainData, validData_new])
        concatLabs = np.concatenate([trainLabels, validLabels_new])
        
        
        ANN_cl = ANN(20600, 7) 
        LSH_ANN_cl = LSH_ANN(80, 15, 5)
        
        start = t.time()        # milliseconds since epoch
        ANN_cl.train(concatData, concatLabs)
        end = t.time()
        print "Took us " + str(end - start) + " milliseconds to train the ANN classifier."
        
        start = t.time() 
        LSH_ANN_cl.train(concatData, concatLabs)
        end = t.time()
        print "Took us " + str(end - start) + " milliseconds to train the LSH_ANN classifier."
        
        # Step 4: test and compare classifiers on testing data.
        
        start = t.time() 
        ANN_cl_acc = ANN_cl.test(testData, testLabels, None)
        end = t.time()
        print "On the testing data, the ANN classifier achieved an accuracy of: " + str(ANN_cl_acc) + "."
        print "It took it " + str(end - start) + " milliseconds to test the data."
        
        start = t.time()
        LSH_ANN_cl_acc = LSH_ANN_cl.test(testData, testLabels)
        end = t.time()
        
        print "On the testing data, the LSH_ANN classifier achieved an accuracy of: " + str(LSH_ANN_cl_acc) + "."
        print "It took it " + str(end - start) + " milliseconds to test the data."
    except Exception as e:
        print "An exception occurred: " + str(e)

        