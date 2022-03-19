
# Standard Libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random

# from fastdtw import fastdtw
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
from collections import Counter

# Machine Learning Libraries
import tensorflow as tf

from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, AveragePooling1D, LSTM
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

from scipy.spatial.distance import euclidean

from numpy.random import seed

seed(10)


def train_MCA_model(X_MC_train, y_train, FAKE_MC):
    
    clf_MC = LogisticRegression(max_iter=1000) # Create Model
    
    if FAKE_MC:
        clf_MC.fit(np.vstack((X_MC_train, X_FAKE_MC)), 
                   np.concatenate((y_train, y_FAKE))) # Train Model
    else:
        clf_MC.fit(X_MC_train, y_train) # Train Model
    clf_MC_proba = clf_MC.predict_proba(X_MC_train) # Output Probabilities
    
    return (clf_MC, clf_MC_proba)
    
    
def train_ACA_model(X_AC_train, y_train, FAKE_AC, NN):
    if NN:
        clf_AC = create_model()
        
        clf_AC.compile(optimizer='adam', 
                        loss='sparse_categorical_crossentropy', 
                        metrics=['accuracy'])
        
        if FAKE_AC:
            X_ = np.vstack((X_AC_train, X_FAKE_AC))
            y_ = np.concatenate((y_train, y_FAKE))
        else:
            X_ = X_AC_train.copy()
            y_ = y_train.copy()
            
        h = clf_AC.fit(X_, y_, shuffle=True, batch_size=256, epochs=200,
                    verbose=0, validation_data=(X_AC_test, y_test))
        
        clf_AC_proba = clf_AC.predict(X_AC_train)
        
        
    else:
        clf_AC = KNeighborsClassifier(n_neighbors=20)
        
        if FAKE_AC:
            clf_AC.fit(np.vstack((X_AC_train, X_FAKE_AC)), 
                       np.concatenate((y_train, y_FAKE)))
        else:
            clf_AC.fit(X_AC_train, y_train)
        
        clf_AC_proba = clf_AC.predict_proba(X_AC_train)
        
    return (clf_AC, clf_AC_proba)
    
    
def train_FFI_model(X_FFI_train, y_train):
    clf_FFI = LogisticRegression(max_iter=1000) # Create Model
    clf_FFI.fit(X_FFI_train, y_train) # Train Model
    clf_FFI_proba = clf_FFI.predict_proba(X_FFI_train) # Output Probabilities
    return (clf_FFI, clf_FFI_proba)
    
        
def train_AMCA_model(X_AC_MC_train, y_train):
#     clf = LogisticRegressionCV(max_iter=1000, fit_intercept=False,
#                                cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0))
    
    clf = LogisticRegression(max_iter=1000, fit_intercept=False)
    
    clf.fit(X_AC_MC_train, y_train)

    clf_proba = clf.predict_proba(X_AC_MC_train)
    
    return (clf, clf_proba)

def create_model():

  comb = Sequential()

  comb.add(Conv1D(filters = 16, kernel_size = 5, activation ='relu', input_shape = (35, 1)))
  comb.add(MaxPooling1D(pool_size=2))
  comb.add(Conv1D(8, 5, activation ='relu'))
  comb.add(Conv1D(8, 5, activation ='relu'))
  comb.add(Dropout(0.5))
  comb.add(LSTM(32, activation ='tanh'))
  comb.add(Dropout(0.5))
  comb.add(Flatten())

  comb.add(Dense(5, activation = 'softmax'))

  return comb

# create_model().summary()

def create_model_2():
  rnn = Sequential()

  rnn.add(LSTM(32, input_shape = (35, 1), activation='tanh'))
  rnn.add(Dropout(0.5))

  rnn.add(Dense(5, activation = 'softmax'))
  
  return rnn

# create_model_2().summary()