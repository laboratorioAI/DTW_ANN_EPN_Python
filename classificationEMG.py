#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 00:07:43 2020

@author: aguasharo
"""


from preProcessing import *
from featureExtraction import *
from postProcessing import *
import pandas as pd
import numpy as np
import time


def classifyEMG_SegmentationNN(dataX_test, centers, model):
    # This function applies a hand gesture recognition model based on artificial
    # feed-forward neural networks and automatic feature extraction to a set of
    # EMGs conatined in the set test_X. The actual label of each EMG in test_X
    # is in the set test_Y. The structure nnModel contains the trained neural
    # network     
       
    window_length = 600
    stride_length = 30
    emg_length = len(dataX_test)
    predLabel_seq = []
    vecTime = []
    timeSeq = []
    
    
    count = 0
    while True:
        start_point = stride_length*count + 1
        end_point = start_point + window_length - 1
        
        if end_point > emg_length:
            break
        
        tStart = time.time()
        window_emg = dataX_test.iloc[start_point:end_point]   
        filt_window_emg = window_emg.apply(preProcessEMGSegment)
        window_sum  = filt_window_emg.sum(axis=1)
        idx_start, idx_end = detectMuscleActivity(window_sum)
        t_acq = time.time()-tStart
        
        if (idx_start != 1) & (idx_end != len(window_emg)) & ((idx_end - idx_start) > 85):
            
            tStart = time.time()
            
            filt_window_emg1 = window_emg.apply(preProcessEMGSegment)
            window_emg1 = filt_window_emg1.iloc[idx_start:idx_end]
            
            
            t_filt = time.time() - tStart
            
            tStart = time.time()
            featVector = featureExtractionf([window_emg1], centers)
            featVectorP = preProcessFeatureVector(featVector)
            t_featExtra =  time.time() - tStart
            
            tStart = time.time()
            x = model.predict_proba(featVectorP).tolist()
            probNN = x[0]
            max_probNN = max(probNN)
            predicted_labelNN = probNN.index(max_probNN) + 1
            t_classiNN = time.time() - tStart
            
            tStart = time.time()
            if max_probNN <= 0.5:
                predicted_labelNN = 1
            t_threshNN = time.time() - tStart 
            #print(predicted_labelNN)
           
        else:
            
            t_filt = 0
            t_featExtra = 0
            t_classiNN = 0
            t_threshNN = 0
            predicted_labelNN = 1
            #print('1')
            
            
        count = count + 1
        predLabel_seq.append(predicted_labelNN)
        vecTime.append(start_point+(window_length/2)+50)
        timeSeq.append(t_acq + t_filt + t_featExtra + t_classiNN + t_threshNN)    
    
    pred_seq = majorite_vote(predLabel_seq, 5, 5)    
        
    return  pred_seq, vecTime, timeSeq




def post_ProcessLabels(predicted_Seq):   
    # This function post-processes the sequence of labels returned by a
    # classifier. Each row of predictedSeq is a sequence of 
    # labels predicted by a different classifier for the jth example belonging
    # to the ith actual class.
    
    time_post = []
    predictions = predicted_Seq.copy()
    predictions[0] = 1
    postProcessed_Labels = predictions.copy()
        
    for i in range(1,len(predictions)):
        
        tStart = time.time()
        
        if predictions[i] == predictions[i-1]:
            cond = 1
        else:    
            cond = 0
            
        postProcessed_Labels[i] =  (1 * cond) + (predictions[i]* (1 - cond))
        t_post = time.time() - tStart
        time_post.append(t_post)
        
    time_post.insert(0,time_post[0])     
    uniqueLabels = unique(postProcessed_Labels)
    
    an_iterator = filter(lambda number: number != 1, uniqueLabels)
    uniqueLabelsWithoutRest = list(an_iterator)
       
    if not uniqueLabelsWithoutRest:
        
        finalLabel = 1
        
    else:
        
        if len(uniqueLabelsWithoutRest) > 1:
            finalLabel = uniqueLabelsWithoutRest[0]
            
        else:
            finalLabel = uniqueLabelsWithoutRest[0]
                   
    
    return finalLabel, time_post









