import numpy as np

from delfi.summarystats.BaseSummaryStats import BaseSummaryStats



def mean_std(repetition_list):
    # See BaseSummaryStats.py for docstring
    
    # get the number of repetitions contained
    n_reps = len(repetition_list)

    # build a matrix of n_reps x 1
    n=0
    M2 = np.zeros(repetition_list[0].shape)
    mean = np.zeros(repetition_list[0].shape)

    # for every repetition, take the mean of the data in the dict
    for x in repetition_list:
        n += 1
        delta = x - mean
        mean += delta/n
        M2 += delta*(x - mean)

    if n < 2:
        return float('nan')
    else:
        return mean, M2/(n-1)
    
    
