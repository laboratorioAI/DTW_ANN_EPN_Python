#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 00:03:12 2020

@author: aguasharo
"""



def majorite_vote(data, before, after):
    # This function is used to apply pos-processing based on majority vote
    
    votes =[0,0,0,0,0,0]
    class_maj = []
        
    for j in range(0,len(data)):
        wind_mv = data[max(0,(j-before)):min(len(data),(j+after))]
        
        for k in range(0, 6):
            a = [1 if i == k+1 else 0 for i in wind_mv]  
            votes[k] = sum(a)
            
        findNumber = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
        idx_label = findNumber(max(votes),votes)
        class_maj.append( idx_label[0] + 1)
        
    
    return class_maj


def unique(list1): 
    # insert the list to the set 
    list_set = set(list1) 
    # convert the set to the list 
    unique_list = (list(list_set)) 
    
    return unique_list 
