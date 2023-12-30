import numpy as np
import math

def rmse(actual, predicted):
    '''
        Compute accuracy by using RMSE

        actual: target values
        predicted: results of training
    '''
    diff = predicted - actual
    squared_diff = diff ** 2
    sum = np.sum(squared_diff)
    frac = sum / actual.shape[0]
    return math.sqrt(frac)
    
