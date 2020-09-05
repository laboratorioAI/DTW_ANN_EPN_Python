#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 23:57:41 2020

@author: aguasharo
"""


from readDataset import *
from preProcessing import *
from featureExtraction import *
from classificationEMG import *



from sklearn.manifold import TSNE
import seaborn as sns


import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.optimizers import Adam, SGD


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


from sklearn.preprocessing import StandardScaler

import collections
from collections import Counter



def get_x_train(user,sample):
    # This function reads the time series(x) of the user (Training Sample)
    train_samples = user['trainingSamples']
    x = (train_samples[sample]['emg'])
    # Divide to 128 for having a signal between -1 and 1
    df = pd.DataFrame.from_dict(x) / 128
    # Apply filter
    train_filtered_X = df.apply(preProcessEMGSegment)
    # Segment the filtered EMG signal
    train_segment_X = EMG_segment(train_filtered_X)
    
    return train_segment_X


            
def get_x_test(user,sample):
    # This function reads the time series(x) of the user (Testing Sample)
    test_samples = user['testingSamples']
    x = (test_samples[sample]['emg'])
    df = pd.DataFrame.from_dict(x) / 128
    
    return df



def decode_targets(y_train):
    # Encode targets to train the Neural Network
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    target = np_utils.to_categorical(encoded_Y)
    
    return target 

    
def get_xy_val(X_train, targets):            
    
    # Get validation data to train Neural Network            
    data_val = X_train.copy()
    data_val['6'] = targets
    
    xy_val = data_val.sample(frac=1).reset_index(drop=True)
    
    
    X_val = xy_val.iloc[:,0:6]  
    y_val = decode_targets(xy_val['6'])
    
    
    return X_val, y_val



def bestCenter_Class(train_segment_X):
    
    # This function returns a set of time series called centers
    # for each gesture class
    
    g1 = train_segment_X[0:25]
    g2 = train_segment_X[25:50]
    g3 = train_segment_X[50:75]
    g4 = train_segment_X[75:100]
    g5 = train_segment_X[100:125]
    g6 = train_segment_X[125:150]
    
    gen = [g1, g2, g3, g4, g5 ,g6]
    
    c = [findCentersClass(g) for g in gen]
             
    return c



def getFeatureExtraction(emg_filtered, centers):
    
    features = featureExtractionf(emg_filtered, centers)  
    dataX = preProcessFeatureVector(features)
    
    return dataX
    
    

def trainFeedForwardNetwork(X_train,y_train, X_test, y_test):
    # This function trains an  artificial feed-forward neural networks 
    # Cost lost: categorical cross entropy
    # Hidden Layer: Tanh
    # Output Layer: softmax
    
    classifier = Sequential()
    
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = None, input_dim = 6))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'tanh'))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    classifier.fit(X_train, y_train, batch_size = 150, epochs = 1500, validation_data = (X_test, y_test), verbose = 0 )
    
    return classifier




def testing_prediction(user,sample,centers,estimator): 
    test_RawX = get_x_test(user,sample) 
    [predictedSeq, vec_time, time_seq]= classifyEMG_SegmentationNN(test_RawX, centers, estimator)
    predicted_label, t_post = post_ProcessLabels(predictedSeq)
    
    # Computing the time of processing
    estimatedTime =  [sum(x) for x in zip(time_seq, t_post)]

    
    return predicted_label, predictedSeq, vec_time, estimatedTime



def recognition_results(results):  
     # This function save the responses of each user into a dictionary

    d = collections.defaultdict(dict)
    
    for i in range(0,150):
                
        d['idx_%s' %i]['class'] = code2gesture(results[i][0])
        d['idx_%s' %i]['vectorOfLabels'] = code2gesture_labels(results[i][1])
        d['idx_%s' %i]['vectorOfTimePoints'] = results[i][2]
        d['idx_%s' %i]['vectorOfProcessingTime']= results[i][3]    
       
    return d




