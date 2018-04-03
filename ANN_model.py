#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 23:08:35 2018

@author: richard
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values  # numpy.array
y = dataset.iloc[:, 13].values

dataset_P = pd.read_csv('predict_HW.csv')
X_predict = dataset_P.iloc[:,3:13].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

X_predict[:,1] = labelencoder_X_1.transform(X_predict[:,1])

labelencoder_X_1 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])
X_predict[:,2] = labelencoder_X_1.transform(X_predict[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]
X_predict = onehotencoder.transform(X_predict).toarray()
X_predict = X_predict[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_predict = sc.transform(X_predict)

# Making ANN

#importing keras
import keras
from keras.models import Sequential
from keras.layers import Dense

# declar 
classifier = Sequential()


# making input and hidden layer
# the first one has to introduce the input layer by "input_dim= (numbers of the independent variables)"
classifier.add(Dense(6, input_dim = 11, activation='relu', bias_initializer='uniform' ))
#constructing a hiden layer
classifier.add(Dense(6, activation='relu', bias_initializer='uniform' ))
#constructing the output layer,  the first arguement(unit) is to find how many output you have, was output_dim
# in classification case, the output layer, we use sigmoid.
classifier.add(Dense(1, activation='sigmoid', bias_initializer='uniform' ))

# how could it improve the wieght, and the loss is to find which kind of output we have, the metrics is the method we use to perform
classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
# this command can train the training set, epochs is how many rounds we should have for whole training set
classifier.fit(X_train, y_train, batch_size=10, epochs= 10)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
y_HWpred = classifier.predict(X_predict)
y_HWpred = (y_HWpred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense



def build_calssifier ():
    classifier = Sequential()
    classifier.add(Dense(6, input_dim = 11, activation='relu', bias_initializer='uniform' ))
    classifier.add(Dense(6, activation='relu', bias_initializer='uniform' ))
    classifier.add(Dense(1, activation='sigmoid', bias_initializer='uniform' ))
    classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
    return classifier
    
classifier = KerasClassifier(build_fn = build_calssifier, batch_size = 10, epochs = 10)
accuracies = cross_val_score(estimator =  classifier, X= X_train, y= y_train, cv =10, n_jobs=1)
accuracies.mean()
accuracies.std()


## Grid search
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense



def build_calssifier (optimizer):
    classifier = Sequential()
    classifier.add(Dense(6, input_dim = 11, activation='relu', bias_initializer='uniform' ))
    classifier.add(Dense(6, activation='relu', bias_initializer='uniform' ))
    classifier.add(Dense(1, activation='sigmoid', bias_initializer='uniform' ))
    classifier.compile(optimizer=optimizer, loss = 'binary_crossentropy', metrics=['accuracy'])
    return classifier
    
classifier = KerasClassifier(build_fn = build_calssifier)
parameters = {'batch_size':[25,32], 
              'nb_epoch':[100,500],
              'optimizer':['adam','rmsprop']
              }

grid_research = GridSearchCV(estimator=classifier,  
                             param_grid = parameters,
                             scoring ='accuracy',
                             cv=10
                             )
grid_research = grid_research.fit(X_train, y_train)
best_parameters = grid_research.best_params_
best_accuracy = grid_research.best_score_