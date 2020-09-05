#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 22:33:37 2020

@author: aguasharo
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import math


def butter_lowpass_filter(data, fs, order):
    # Get the filter coefficients 
    b, a = signal.butter(order, fs, 'low', analog = False)
    y = signal.filtfilt(b, a, data)
    return y



def preProcessEMGSegment(EMGsegment_in):
    # This function to apply a filter
    EMG = max(EMGsegment_in)
    
    if EMG > 1:
        EMGnormalized = EMGsegment_in/128
    else:
        EMGnormalized = EMGsegment_in    
             
    EMGrectified = abs(EMGnormalized)   
    EMGsegment_out = butter_lowpass_filter(EMGrectified, 0.1, 5) 
    return EMGsegment_out



def detectMuscleActivity(emg_sum): 
    # This function segments in a EMG the region corresponding to a muscle
    # contraction. The indices idxStart and idxEnd correspond to the begining
    # and the end of such a region

    # Sampling frequency of the EMG
    fs = 200
    minWindowLength_Segmentation =  100 # Minimum length of the segmented region
    hammingWdw_Length = np.hamming(25) # Window length
    numSamples_lapBetweenWdws = 10 # Overlap between 2 consecutive windows
    threshForSum_AlongFreqInSpec = 0.86

    [s, f, t, im] = plt.specgram(emg_sum, NFFT = 25, Fs = fs, window = hammingWdw_Length, noverlap = numSamples_lapBetweenWdws, mode = 'magnitude', pad_to = 50)  
    
    # Summing the spectrogram along the frequencies
    sumAlongFreq = [sum(x) for x in zip(*s)]

    greaterThanThresh = []
    # Thresholding the sum sumAlongFreq
    for item in sumAlongFreq:
        if item >= threshForSum_AlongFreqInSpec:
            greaterThanThresh.append(1)
        else:
            greaterThanThresh.append(0)
           
    greaterThanThresh.insert(0,0)       
    greaterThanThresh.append(0)    
    diffGreaterThanThresh = abs(np.diff(greaterThanThresh)) 

    if diffGreaterThanThresh[-1] == 1:
        diffGreaterThanThresh[-2] = 1;      
       
    x = diffGreaterThanThresh[0:-1];
    findNumber = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    idxNonZero = findNumber(1,x)
    numIdxNonZero = len(idxNonZero)
    idx_Samples = np.floor(fs*t)
    # Finding the indices of the start and the end of a muscle contraction
    if numIdxNonZero == 0:
        idx_Start = 1
        idx_End = len(emg_sum)
    elif numIdxNonZero == 1:
        idx_Start = idx_Samples[idxNonZero]
        idx_End = len(emg_sum)
    else:
        idx_Start = idx_Samples[idxNonZero[0]]
        idx_End = idx_Samples[idxNonZero[-1]-1]
    # Adding a head and a tail to the segmentation
    numExtraSamples = 25
    idx_Start = max(1,idx_Start - numExtraSamples)
    idx_End = min(len(emg_sum), idx_End + numExtraSamples)
    
    if (idx_End - idx_Start) < minWindowLength_Segmentation:
        idx_Start = 1
        idx_End = len(emg_sum)


    return int(idx_Start), int(idx_End)


def EMG_segment(train_filtered_X):
    # This function return a segment with corresponding to a muscle
    # contraction
    
    df_sum  = train_filtered_X.sum(axis=1)
    idx_Start, idx_End = detectMuscleActivity(df_sum)
    df_seg = train_filtered_X.iloc[idx_Start:idx_End]
    
    return df_seg 