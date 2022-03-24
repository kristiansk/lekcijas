# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

#np.set_printoptions(supress=True)

(X_train, y_train), (X_test, y_test) = np.load("mnist.npy", allow_pickle=True)

print(np.shape(y_test))


X_train = np.reshape(X_train, (X_train.shape[0], -1, 1))
X_test = np.reshape(X_test, (X_test.shape[0], -1, 1))
y_train = np.reshape(y_train, (6000,10,1))
y_test = np.reshape(y_test, (1000,10,1))

#%%

X_train = X_train/255
X_test = X_test/255




#%%



class Neural_Network():
    
    def __init__(self. sizes):
        
        self.n_layers = len(sizes)
        self.layer_sizes = sizes #[784, 30, 10]
        self.biases = [np.random.normal(size = (y, 1)) for y in sizes[1:]]
        
        
        
        
