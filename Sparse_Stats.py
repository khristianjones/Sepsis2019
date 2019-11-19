#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code to determine the sparsity of the data
Created on Tue Nov  5 17:40:24 2019
@author: Erik
"""

###############################################################################
import numpy as np
import os.path
import functions
###############################################################################



NUM_FEAT = 41   #change if feature size changes


###############################################################################
#Get all the data stacked up
train_listA, test_listA, train_listB, test_listB = functions.get_train_test()

patient_list = np.zeros(41)

patient_number = 1
for file_name in train_listA:
        
        
        print("\r working patient for train list A: "+ str(patient_number), end= '')
        
        current_file_name = "p{0:06d}.psv".format(file_name)
        file_to_open = os.path.join("Training/Training2/trainingA/training/", current_file_name) #remove /home/khristian/Documents/ before submitting, and add the training files to the folder"
        
        ICU_values, column_names = read_challenge_data(file_to_open)
        patient_list= np.vstack((patient_list, ICU_values))
        patient_number += 1
        
patient_number = 1       
for file_name in train_listB:
        
        
        print("\r working patient from train list B: "+ str(patient_number), end= '')
        
        current_file_name = "p{0:06d}.psv".format(file_name)
        file_to_open = os.path.join("/home/khristian/Documents/Training/Training2/trainingB/training_setB/", current_file_name) #remove /home/khristian/Documents/ before submitting, and add the training files to the folder"
        
        ICU_values, column_names = read_challenge_data(file_to_open)
        patient_list= np.vstack((patient_list, ICU_values))
        patient_number += 1
        
patient_number = 1     
for file_name in test_listA:
        
        
        print("\r working patient from test list A: "+ str(patient_number), end= '')
        
        current_file_name = "p{0:06d}.psv".format(file_name)
        file_to_open = os.path.join("Training/Training2/trainingA/training/", current_file_name) #remove /home/khristian/Documents/ before submitting, and add the training files to the folder"
        
        ICU_values, column_names = read_challenge_data(file_to_open)
        patient_list= np.vstack((patient_list, ICU_values))
        patient_number += 1
        
patient_number = 1      
for file_name in test_listB:
        
        
        print("\r working patient from test list B: "+ str(patient_number), end= '')
        
        current_file_name = "p{0:06d}.psv".format(file_name)
        file_to_open = os.path.join("/home/khristian/Documents/Training/Training2/trainingB/training_setB/", current_file_name) #remove /home/khristian/Documents/ before submitting, and add the training files to the folder"
        
        ICU_values, column_names = read_challenge_data(file_to_open)
        patient_list= np.vstack((patient_list, ICU_values))
        patient_number += 1
        
        
        
###############################################################################
        
# =============================================================================
# current_file_name = "p019722.psv"
# file_to_open = os.path.join("/home/khristian/Documents/Training/Training2/trainingA/training", current_file_name)
# 
# patient_list, column_names = read_challenge_data(file_to_open)
# =============================================================================

#For loop to determine how many values are contained in each feature
i = 0   #feature iterator
j = 0   #hour iterator
amount_filled = np.zeros(NUM_FEAT)


for i in range(NUM_FEAT):
    for j in range(len(patient_list)):
        test_value = patient_list[j, i]
        
        if float('-inf') < float(test_value) < float('inf'):
            amount_filled[i] += 1
            
percent_filled = (amount_filled / len(patient_list))*100
    
    
    
    
    
    
    
    
    
    
    
    
    
    