#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 22:10:32 2019

@author: khristian
"""

import os
import os.path

import matplotlib
matplotlib.rcParams["backend"] = "TkAgg"
import matplotlib.pyplot as plt
import numpy as np
#filename = os.path.join(dir, 'relative','path','to','file','you','want')
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

#####################Creates train test for the first file, I believe this only works on windows###################
cwd = os.getcwd()
counter = 0
counter2 = 0
for filename in os.listdir("input_directory"):
    counter +=1
    filename1 = os.path.join(cwd, 'input_directory', filename)
    
    with open(filename1) as fp:
        for line in fp:
            line = fp.readline()
            line = line.strip()
            if len(line) > 0:
                if line[-1] == '1':
                    indicator1 = 1
                    break
                else:
                    indicator1 = 0
                    continue
            else:
                continue
    second_file = os.path.join(cwd, 'output_directory', filename)   
    with open(second_file) as sf:
         for line in sf:
             line = sf.readline()
             line = line.strip()
             
             if len(line) > 0:
                if line[-1] == '1':
                    indicator2 = 1
                    break
                else:
                    indicator2 = 0
                    continue
             else:
                continue
             counter2 +=1
    if indicator1==1 and indicator2==1:
        true_positive +=1
    elif indicator1==0 and indicator2==0:
        true_negative +=1
    elif indicator1==0 and indicator2==1:
        false_positive +=1
    elif indicator1==1 and indicator2==0:
        false_negative +=1    
fp.close()
print(true_positive, true_negative, false_positive, false_negative) 
print(counter, counter2)   

positive = true_positive + false_positive
negative = true_negative + false_negative

true_positive = true_positive/positive
true_negative = true_negative/negative
false_positive = false_positive/positive
false_negative = false_negative/negative

true_positive = true_positive*100
true_negative = true_negative*100
false_positive = false_positive*100
false_negative = false_negative*100

print(true_positive, true_negative, false_positive, false_negative)


sepsis = np.array([true_positive, false_positive])
non_sepsis = np.array([true_negative, false_negative])
xpos = np.array([0, 1])
sep_objects = ('True\nPositive', 'False \nPositive')
non_objects = ('True \nNegative', 'False \nNegative')

plt.bar(xpos, sepsis, 0.5, align='center', alpha=0.5)

plt.title('Sepsis Patient Prediction Accuracy')
plt.ylabel('percent ')
plt.xticks(xpos, sep_objects)
plt.axis([-0.5, 1.5, 0, 100], 'scaled')  
plt.show()

# =============================================================================
# plt.bar(xpos, non_sepsis, 0.5, align='center', alpha=0.5)
# 
# plt.title('Non-Sepsis Patient Prediction Accuracy')
# plt.ylabel('percent ')
# plt.xticks(xpos, non_objects)
# plt.axis([-0.5, 1.5, 0, 100], 'scaled')  
# plt.show()
# 
# =============================================================================


