# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 12:27:32 2022

@author: kk16031
"""
import numpy as np
from tensorflow import keras

np.set_printoptions(suppress=True)

(X_train, y_train), (X_test, y_test) = np.load('mnist.npy', allow_pickle=True)


y_train = keras.utils.to_categorical(y_train)

X_train = X_train/255
X_test = X_test/255

inp = 784

slani = [[40],  [20, 20], [10, 10, 10, 10], [30, 10], [10, 30]]


model = keras.model.Sequential()
model.add(keras.layers.flatten(input_shape = [28, 28]))
for i in slani:
    
    for j in i:
        model.add(keras.layers.Dense(j, activation="sigmoid"))
        
        
        
        
    model.summary()
    model.compile(loss="mse", metrics=['accuracy'], optimizer="sgd")
    
    history = model.fit(X_train, y_train, batch_size = 100, epochs = 5)
    
    
    
    
    





