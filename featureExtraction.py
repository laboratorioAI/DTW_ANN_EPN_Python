#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 23:22:24 2020

@author: aguasharo
"""

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def findCentersClass(emg_filtered):
    # This function returns a set of time series called centers. The ith
    # time series of centers, centers{i}, is the center of the cluster of time 
    # series from the set timeSeries that belong to the ith class. For finding
    # the center of each class, the DTW distance is used.  

    distances = []
    sample = 25
    column = np.arange(0,sample)
    mtx_distances = pd.DataFrame(columns = column)
    mtx_distances = mtx_distances.fillna(0) # with 0s rather than NaNs
    
    
    for sample_i in emg_filtered:
        for sample_j in emg_filtered:   
            dist, dummy = fastdtw(sample_i, sample_j, dist = euclidean)
            distances.append(dist)
            
        df_length = len(mtx_distances)
        mtx_distances.loc[df_length] = distances 
        distances= []  
    vector_dist = mtx_distances.sum(axis=0)
    idx = vector_dist.idxmin()
    center_idx = emg_filtered[int(idx)]
    
    return center_idx


def featureExtractionf(emg_filtered, centers):
    # This function computes a feature vector for each element from the set
    # timeSeries. The dimension of this feature vector depends on the number of 
    # time series of the set centers. The value of the jth feature of the ith
    # vector in dataX corresponds to the DTW distance between the signals 
    # timeSeries{i} and centers{j}.  

    dist_features = []
    
    column = np.arange(0,len(centers))
    dataX = pd.DataFrame(columns = column)
    dataX = dataX.fillna(0)
    
    for rep in emg_filtered:
        for middle in centers:
            dist, dummy = fastdtw(rep, middle, dist = euclidean) 
            dist_features.append(dist)
        
        dataX_length = len(dataX)
        dataX.loc[dataX_length] = dist_features
        dist_features = [] 
    
    return dataX


def preProcessFeatureVector(dataX_in):
    # This function preprocess each feature vector of the set dataX_in. Each
    # row of dataX_in is a fetaure vector and each column is a featur
    num_gestures = 6
    dataX_mean = dataX_in.mean(axis = 1)
    dataX_std = dataX_in.std(axis = 1)   
    dataX_mean6 =  pd.concat([dataX_mean]*num_gestures, axis = 1)
    dataX_std6 =  pd.concat([dataX_std]*num_gestures, axis = 1)   
    dataX6 = (dataX_in - dataX_mean6)/dataX_std6
    
    return dataX6
