#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 23:59:24 2020

@author: aguasharo
"""

from recognitionFunctions import *


class RecognitionModel:
    
    num_gestures = 6
    
    def __init__(self,version,user):
        self.user = user
        self.version = version
        
     
    def preProcessingData(self):
        sample_type = self.version+'Samples'
        # Reading the training samples
        train_samples = self.user[sample_type]
        # Preprocessing
        train_segment_X = [get_x_train(self.user,sample) for sample in train_samples] 
        
        return train_segment_X 
   
    def featureExtraction(self, train_data):         
        # Finding the EMG that is the center of each class
        centers = bestCenter_Class(train_data)  
        # Feature extraction by computing the DTW distance between each training
        # example and the center of each cluster           
        # Preprocessing the feature vectors    
        X_train = getFeatureExtraction(train_data, centers)
         
        return X_train, centers
         
         
    def trainSoftmaxNN(self, X_train):
        sample_type = self.version+'Samples'
        # Reading the training samples
        train_samples = self.user[sample_type]      
        # Training the feed-forward NN
        y_train = decode_targets(get_y_train(train_samples)) 
        X_val, y_val = get_xy_val(X_train, get_y_train(train_samples)) 
        estimator = trainFeedForwardNetwork(X_train, y_train, X_val, y_val)          
        
        return estimator
       
      
    def classifyGestures(self,version, estimator, centers) :
        
        sample_type = self.version+'Samples'
        # Reading the testing samples    
        test_samples = self.user[sample_type]      
        # Concatenating the predictions of all the users for computing the
        # errors
        response = ([testing_prediction(self.user, sample, centers, estimator) for sample in test_samples]) 
        
        results = recognition_results(response)
        
        return results
    