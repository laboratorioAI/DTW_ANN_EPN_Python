#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 22:37:27 2020

@author: aguasharo
"""

from __future__ import print_function

import json
import os
import collections

from RecognitionModel import RecognitionModel
        




#%% Read user data
response = collections.defaultdict(dict)
num_gestures = 6
folderData = 'testingJSON'
cont = 0
entries = os.listdir(folderData)
num_users = len(entries)


for entry in entries:
    cont = cont + 1
    print('Processing data from user: ' + str(cont) + ' / '+ str(num_users))
    
    file_selected = folderData + '/' + entry + '/' + entry + '.json'
    
    with open(file_selected) as file:
        
        # Read user data
        user = json.load(file)      
        name_user = user['userInfo']['name']

        currentUser = RecognitionModel('training', user)     
        # Preprocessing
        train_segment_X  = currentUser.preProcessingData()
        
        # Feature extraction by computing the DTW distance between each training
        # example and the center of each cluster  
        [X_train, centers] = currentUser.featureExtraction(train_segment_X)
        
        # Training the feed-forward NN
        estimator = currentUser.trainSoftmaxNN(X_train)
        
        results = currentUser.classifyGestures('testing', estimator, centers)    
     
     # Concatenating the predictions of all the users for computing the
     # errors    
    response[name_user]['testing'] = results

           
with open('responses.json', 'w') as json_file:
  json.dump(response, json_file)             

































             





                


            

        

           




            



