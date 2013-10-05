'''
Created on Sep 22, 2012

@author: Jason
'''

class DatasetError(Exception):
    '''
    I will use this class to raise dataset-related exceptions.
    Common errors: Empty datasets provided, datasets with wrong
    number of features, datasets lacking some feature or labels, etc
    Used the ready-baked template on the Python tutorial:
        http://docs.python.org/tutorial/errors.html (section 8.5)
    '''
    def __init__(self, description):
        '''
        Constructor
        '''
        self.description = description
        
    def __str__(self):
        return repr(self.description)
        
        
class LogicalError(Exception):
    '''
    Used for more logic-oriented errors, such as providing parameters that make no sense
    for the application at hand.
    '''
    
    def __init__(self, description):
        '''
        Constructor
        ''' 
        self.description = description
        
    def __str__(self):
        return repr(self.description)