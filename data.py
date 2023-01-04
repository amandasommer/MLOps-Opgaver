import torch
import numpy as np


def mnist():
    # exchange with the corrupted mnist dataset
    y=1
    train = np.load('test.npz') #Hov
    test = np.load('test.npz') 
    return train, test

