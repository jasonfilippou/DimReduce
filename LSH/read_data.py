'''
Created on Mar 23, 2013

@author: Jason
'''

import numpy as np
import pickle as pkl
import os


def read_data(file):
    '''
    Reads data from a file. Patterns in rows, features in columns
    @param file an ASCII text file
    @return: a numpy 2D array with patterns in rows.
    '''
    return np.genfromtxt(file, delimiter=' ')
    
def read_labels(file):
    '''
    Reads labels from a file. One label per line.
    '''
    return np.genfromtxt(file, delimiter = ' ')

if __name__ == '__main__':
    
    ''' 
    We need to read the input data and validation data
    and concatenate each example label to the example 
    itself. 
    '''
    input_data_dir = os.listdir('input_data/')
    for f in input_data_dir:
        print "Examining file " + str(f) 
        if f == "gisette_train.data":
            trainDat = read_data('input_data/' + f)
            print "Shape of training data: " + str(np.shape(trainDat))
        elif f =="gisette_valid.data":
            validDat = read_data('input_data/' + f)
            print "Shape of validation data: " + str(np.shape(validDat))
        elif f == "gisette_train.labels":
            trainLabs = read_labels('input_data/' + f)
            print "Shape of training labels: " + str(np.shape(trainLabs))
        elif f == "gisette_valid.labels":
            validLabs = read_labels('input_data/' + f)
            print "Shape of validation labels: " + str(np.shape(validLabs))
       
    # Dump read objects to disk
    pkl.dump(trainDat, open('proc_data/trainDat.pkl', 'wb'))
    pkl.dump(trainLabs, open('proc_data/trainLabs.pkl', 'wb'))
    pkl.dump(validDat, open('proc_data/validDat.pkl', 'wb')) 
    pkl.dump(validLabs, open('proc_data/validLabs.pkl', 'wb'))   
    
    print "Done!"
        