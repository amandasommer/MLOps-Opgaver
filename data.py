import torch
import numpy as np


def mnist():
    # exchange with the corrupted mnist dataset
    x=2
    train = np.load('test.npz') #add
    test = np.load('test.npz') 
    return train, test

