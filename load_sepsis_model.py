###############new version as of June 10 2019###############
#
#This is the part of the algorithm that is only meant to run once
#
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import RMSprop

import numpy as np
import os.path
import math
import time
import warnings

import functions
warnings.filterwarnings('ignore') #Removes warnings from preprocessing scale

#####Global Variables##########
#PATH = '/home/khristian/Desktop/Summer Research 201/Training2/trainingA/training/p000001.psv'
patient_list= [1, 2]
#####Main Program##############
def read_challenge_data(input_file):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_names = np.array(header.split('|'))
        values = np.loadtxt(f, delimiter='|')
# =============================================================================
#     # ignore SepsisLabel column if present #use if you need to ignore sepsis label
#     if column_names[-1] == 'SepsisLabel':
#         column_names = column_names[:-1]
#         values = values[:, :-1]
# =============================================================================
    return (values, column_names)
def hour_by_hour(patient):
    current_matrix = patient[0, :]
    #sepsis_label = patient[0, -1]
    current_matrix = np.nan_to_num(current_matrix)
    #Preprocesses the data. fills NaN's with whatever the previous value was

    for hour in range(len(patient)):
        if hour == 0:
             current_matrix = patient[0, :]
             current_matrix = np.nan_to_num(current_matrix)
             
             #sepsis_label = patient[0, -1]
        else:    
            current_hour = patient[hour, :]
            current_hour = np.nan_to_num(current_hour)
            current_matrix = np.vstack((current_matrix, current_hour)) #Stacks current hour on
            
            #current_label = [hour, -1]
            #sepsis_label = np.vstack((sepsis_label, current_label)) #Stacks current hour on
            #Preprocesses the data. fills with mean values of columns
            for feature in range(len(current_matrix[0,:])):
                if current_matrix[hour, feature] == 0:
                    current_matrix[hour, feature] = current_matrix[hour-1, feature]
                else:
                    continue
    sepsis_labels = current_matrix[:, -1:]
    current_matrix = np.delete(current_matrix, 40, 1)
    current_matrix = scale(current_matrix, axis=0, with_mean=True, with_std=True, copy=True)
    current_matrix[:, -1:] = sepsis_labels
    
    one_hot_labels = np.zeros((len(sepsis_labels),2))
    i=0
    for i in range(len(sepsis_labels)):
        if sepsis_labels[i] == 0:
            one_hot_labels[i, :] = [1, 0] 
        else:
            one_hot_labels[i, :] = [0, 1]

    return(current_matrix, sepsis_labels, one_hot_labels)



train_listA, test_listA, train_listB, test_listB = functions.get_train_test()
patient_number = 1  #counter for printing what patient is being worked

#Creates the Stacked Training list
start = time.time()
stacked_train = np.zeros(40)
stacked_labels = np.zeros(1)
stacked_one_hot = np.zeros(2)
for file_name in train_listB:
        
        
        print("\r working patient: "+ str(patient_number), end= '')
        
        current_file_name = "p{0:06d}.psv".format(file_name)
        file_to_open = os.path.join("/home/khristian/Documents/Training/Training2/trainingB/training_setB/", current_file_name) #remove /home/khristian/Documents/ before submitting, and add the training files to the folder"
        
        ICU_values, column_names = read_challenge_data(file_to_open)
        current_patient, sepsis_labels, one_hot_labels = hour_by_hour(ICU_values)
        stacked_train = np.vstack((stacked_train, current_patient))
        stacked_labels = np.vstack((stacked_labels, sepsis_labels))
        stacked_one_hot = np.vstack((stacked_one_hot, one_hot_labels))
        patient_number += 1

stacked_train = np.delete(stacked_train, 0, 0)
pca_stacked_train = np.delete(stacked_train, 0, 0)
stacked_labels = np.delete(stacked_labels, 0, 0)
stacked_one_hot = np.delete(stacked_one_hot, 0, 0)
#PCA Transforms the Stacked Training List
pca = PCA(n_components=10)
pca.fit(stacked_train)
    
pca_stacked_train = pca.transform(stacked_train) 

#Adds the labels back on
pca_stacked_train = np.hstack((pca_stacked_train, stacked_one_hot))    
end = time.time()
total = end-  start
print("\t\tstacking took " + str(total) + " seconds\n")    
 
#Deletes initialized row


#Dummy Test Set
current_file_name = "p019722.psv".format(file_name)
file_to_open = os.path.join("/home/khristian/Documents/Training/Training2/trainingA/training", current_file_name)

ICU_values, column_names = read_challenge_data(file_to_open)
test_patient, test_label, test_one_hot = hour_by_hour(ICU_values)

pca_test = pca.transform(test_patient)  
##################Start of Tenserflow #########################################
xx = stacked_one_hot
#xx = tf.cast(xx,tf.float32)
x_train = keras.utils.to_categorical(xx, 2)   #for training

 
 
print(tf.VERSION)
print(tf.keras.__version__)
 
 
 
 
#Building Tensorflow Neural Network 
model = keras.Sequential()
model.add(Dense(6, input_shape=(10,), activation='sigmoid'))
model.add(Dense(4,activation='sigmoid'))
model.add(Dense(2,activation='sigmoid'))


model.compile(optimizer=RMSprop(), 
             loss='categorical_crossentropy',
             metrics=['accuracy'])
 
 
 
model.fit(pca_stacked_train[:,:-2], xx, batch_size=32, epochs=5, shuffle=True, validation_split=0.1)
 
model.summary()    
    
output=model.predict(pca_test[:,:])    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    