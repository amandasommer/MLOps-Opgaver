import torch
import numpy as np


def mnist():
    # exchange with the corrupted mnist dataset
    train = np.load('test.npz') 
    test = np.load('test.npz') 
    return train, test

