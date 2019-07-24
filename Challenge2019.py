#!/usr/bin/python3.6
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import RMSprop

from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.decomposition import PCA

import os.path
import sys
import scipy.io as sio
import functions
#################### Takes Command Line Argument ##############################
# =============================================================================
# if __name__ == '__main__':
#     if len(sys.argv) != 2:
#         sys.exit('Usage: %s input[.psv]' % sys.argv[0])
# 
#     record_name = sys.argv[1]
#     if record_name.endswith('.psv'):
#         record_name = record_name[:-4]
# 
#     # read input data
#     input_file = record_name + '.psv'
# =============================================================================
    

input_file = 'p02911.psv'

############################ Reads in train/test split ########################

mat_contents=sio.loadmat('labels.mat')
abbys_code=sio.loadmat('final_array.mat')

################### Get Train Test Split read in ##############################

train_list, test_list = functions.get_train_test() # can ignore test list for single input

################### Stacks all patients for PCA ###############################

initial_train = functions.sort_train(train_list) 

pca = PCA(n_components=10)
pca.fit(initial_train)
pcomponents = pca.components_

pca_initial = pca.transform(initial_train)

#################### Pulls numpy arrays from .npy files in CWD#################

training_output = np.load('large_patient_matrix.npy')

test_output = functions.test_transform(pca, pcomponents, input_file) #For reading in one test patiet

#test_output = functions.patient_matrix(test_list, pca, pcomponents,  mat_contents, 'label_valid') # for reading in entire test set
####### Stacks all of them on top of eachother and deletes the zero rows#######

stacked_train = functions.stack_windows(training_output)

stacked_test = functions.stack_windows_test(test_output) #For reading in one test patient
#stacked_test = functions.stack_windows(test_output)     #For reading in entire test set
################################################################################


##################Start of Tenserflow #########################################
xx=functions.scale_label(stacked_train)
yy=functions.scale_label(stacked_test)
x_train=keras.utils.to_categorical(xx, 23)   #for training
y_test=keras.utils.to_categorical(yy, 23)    #for testing
 
 
print(tf.VERSION)
print(tf.keras.__version__)
 
 
 
 
#Building Tensorflow  
model = keras.Sequential()
model.add(Dense(128, input_shape=(264,), activation='relu'))
model.add(Dense(64,activation='tanh'))
model.add(Dense(23,activation='softmax'))


model.compile(optimizer=RMSprop(), 
             loss='categorical_crossentropy',
             metrics=['accuracy'])
 
 
 
model.fit(stacked_train[:,0:264], x_train, batch_size=32, epochs=15, shuffle=True, validation_split=0.1)
 
model.summary()
 
output=model.predict(stacked_test[:,0:264])
a=output.shape[0]
b=output.shape[1]
f=open("Answers.txt", "a")
print("Hour|Probability|PredidctedLabel")
past_point=False
count=0

#Calculating the thresholds for calculating percentages
std = np.std(output, axis=0)
mean = np.mean(output, axis=0)
threshold = mean - (std/5)
threshold[0] = 0.99
threshold[20]= mean[20]
percentages = np.array([0.005, 0.005, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.77, 0.80, 0.82, 0.85, 0.87, 0.9, 0.92, 0.95, 0.97, 0.99])
tts = np.array([0, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10])
first_window = output[0]

cur_best_score = first_window[0]*percentages[0]
cur_best_index = 0
for i in range(1, len(first_window)):
    test_score = first_window[i]*percentages[i]
    if first_window[i] > threshold[i] and test_score > cur_best_score:
        cur_best_index = i
        cur_best_score = first_window[i]*percentages[i]

if len(output) == 1:
    first_day = np.zeros(24)
    if tts[cur_best_index] == 0 or tts[cur_best_index] > 0:
        first_day = first_window*percentages
    elif tts[cur_best_index] < 0:
        hour_of_sepsis = len(first_window)+(tts[cur_best_index]+1)
        for i in range((len(first_window)+1)-hour_of_sepsis):
            first_day[i] = first_window[hour_of_sepsis+i]*percentages[hour_of_sepsis+i]
else:
    first_day = np.zeros(24)
    
    if tts[cur_best_index] == 0 or tts[cur_best_index] > 0:
        first_day = first_window*percentages
    elif tts[cur_best_index] < 0:
        hour_of_sepsis = 24+(tts[cur_best_index]+1)
        
        for i in range(24-hour_of_sepsis):
            print(i)
            first_day[i] = first_window[hour_of_sepsis+i]*percentages[hour_of_sepsis+i]
            
    if len(output) > 1:
        for j in range(1, len(output)):
            cur_best_score = output.item((j,0))*percentages.item((0))
            
            for k in range(0, len(output[j,:])):
                piq = output.item((j,k))
                test_score = piq*percentages.item((k))
                if piq > threshold.item((k)) and test_score > cur_best_score:
                    cur_best_index = k
                    cur_best_score = piq*percentages[k]
            first_day = np.hstack((first_day, cur_best_score))


first_day = first_day*100
for l in range(0,len(first_day)):
    f.write(str(count+1))
    f.write("|")
    f.write('{0:.4f}'.format(first_day[l]))
    if(first_day[l]>0.5):
        f.write("\t1")
        past_point=True
    elif(past_point==True):
        f.write("\t1")
    else:
        f.write("\t0")
    f.write("\n")
    count+=1 
f.write(str(count+1))
f.write("|")
f.write('{0:.4f}'.format(first_day[l]))
if(first_day[l]>0.5):
    f.write("\t1")
    past_point=True
elif(past_point==True):
    f.write("\t1")
else:
    f.write("\t0")
f.write("\n")

f.close()
    




