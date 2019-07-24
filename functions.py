#new version as of June 10 2019
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.decomposition import PCA
import numpy as np
import os.path



#when preprocessing, the function does not like when the SD is close to zero,
#so the following ignores the warnings
import warnings
warnings.filterwarnings('ignore') 

def sort_train(train_list):         #also normalizes and fills NaN w/ zero
    
    input_training_data = np.zeros(40)
    patient_number = 1
    for file_name in train_list:
        
        print("\r stacking patient: "+ str(patient_number), end='')
        patient_number += 1
        current_file_name = "p{0:05d}.psv".format(file_name)
        file_to_open = os.path.join("training/", current_file_name)
        
        temp = read_challenge_data(file_to_open)
        imputer = SimpleImputer(missing_values=np.nan, fill_value=None, strategy='constant')
        imputer.fit(temp)
        
        filled_temp = imputer.transform(temp)
        temp_scaled = preprocessing.scale(filled_temp)
        input_training_data = np.vstack((input_training_data, temp_scaled))
        input_training_data = np.delete(input_training_data, 0, 0)  #deletes initialized array
        
    return(input_training_data)       

#Reads in and makes a list with the Train and Test .txt file    
def get_train_test():
    with open("trainA.txt", 'r') as file:
    
        for line in file:
            train_string = line
            train_listA = train_string.split('\t')
        train_listA = list(map(int, train_listA))
    
    file.close()
    
    with open("testA.txt", 'r') as file:
    
        for line in file:
            test_string = line
            test_listA = test_string.split('\t')
        test_listA = list(map(int, test_listA))
    
    file.close()
    
    with open("trainB.txt", 'r') as file:
    
        for line in file:
            train_string = line
            train_listB = train_string.split('\t')
        train_listB = list(map(int, train_listB))
    
    file.close()
    
    with open("testB.txt", 'r') as file:
    
        for line in file:
            test_string = line
            test_listB = test_string.split('\t')
        test_listB = list(map(int, test_listB))
    
    file.close()
    return(train_listA, test_listA, train_listB, test_listB)
    
#reads in a chalenge file and ignores the sepsis label
def read_challenge_data(input_file):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_names = np.array(header.split('|'))
        values = np.loadtxt(f, delimiter='|')
    # ignore SepsisLabel column if present #use if you need to ignore sepsis label
    if column_names[-1] == 'SepsisLabel':
        column_names = column_names[:-1]
        values = values[:, :-1]
    return (values) #insert ", column_names" if you want the column names

#Reads in challenge file and includes the sepsis label
def Sread_challenge_data(input_file):
    with open(input_file, 'r') as f:
        header = f.readline().strip()
        column_names = np.array(header.split('|'))
        values = np.loadtxt(f, delimiter='|')

    return (values) #insert ", column_names" if you want the column names

def find_error(input_data, pca_input, pcomponents):
    
    running_total = np.zeros(40)    #Initializes arrays
    error = np.zeros(40)
    
    for i in range(len(input_data)):
         
        for j in range(10):
            error_temp = pca_input[i,j] * pcomponents[j]
            running_total = np.add(error_temp, running_total)
            
        temp = np.subtract(input_data[i], running_total)
        error = np.vstack((error, temp))
    error = np.delete(error, 0, 0)  #deletes initialized array
    error_mean = np.mean(error, axis = 1)
    error_mean = np.reshape(error_mean, (len(input_data),1))
    return(error_mean) #currently doesn't output plain ol error

#Reads in a list and of file names and creates a transformed matrix with a
#sliding window 
def patient_matrix(train_list, pca, pcomponents,  mat_contents, dic_label):
    MAX_HOURS = 370 #Maximum number of hours a patient can have.
    TOT_FEAT = 11   #number of features = PCA components + error
    TOT_LEN = TOT_FEAT*24   #Total length of the rows = total features * 24 hour window
    patient_number = 0  #counter for printing what patient is being worked
    
    tts_matrix = append_tts(mat_contents, train_list, dic_label)
    output_matrix = np.zeros((MAX_HOURS,TOT_LEN))   #initializes array, gets deleted later
    
    for file_name in train_list:
        input_training_data = np.zeros(40)
        
        print("\r working patient: "+ str(patient_number+1), end= '')
        
        
        current_file_name = "p{0:05d}.psv".format(file_name)
        file_to_open = os.path.join("training/", current_file_name)
        
        temp = read_challenge_data(file_to_open)
        
        imputer = SimpleImputer(missing_values=np.nan, fill_value=None, strategy='constant')
        imputer.fit(temp)
        
        temp = imputer.transform(temp)
        temp = preprocessing.scale(temp)
        input_training_data = np.vstack((input_training_data, temp))
        input_training_data = np.delete(input_training_data, 0, 0)  #deletes initialized array
        
        pca_initial = pca.transform(input_training_data)
        
        error = find_error(input_training_data, pca_initial, pcomponents)
        
        hours = np.hstack((pca_initial, error))
        hour_matrix = np.zeros((MAX_HOURS, TOT_FEAT))
        hour_matrix[0:len((hours)+1),:] = hours
        i = j = 0
        patient = np.zeros((MAX_HOURS, TOT_LEN))
        
        for i in range(len(hours)+1):
            row = np.zeros(TOT_LEN)
            k = 0
            
            for j in range(24):
                row[k:k + TOT_FEAT] = hour_matrix[j+i,:]

                k = ((j + 1) * TOT_FEAT)
            patient[i,:] = row

        output_matrix = np.dstack((output_matrix, patient))
        patient_number += 1
    output_matrix = np.delete(output_matrix, 0, 2)
    output_matrix = np.hstack((output_matrix, tts_matrix))  
    return(output_matrix)



def test_transform(pca, pcomponents, current_file_name):
    MAX_HOURS = 370 #Maximum number of hours a patient can have.
    TOT_FEAT = 11   #number of features = PCA components + error
    TOT_LEN = TOT_FEAT*24   #Total length of the rows = total features * 24 hour window
    
    input_training_data = np.zeros(40)    
    
    #Depending how we format the input file we will need this
    #file_to_open = os.path.join("training/", current_file_name)
    
    temp = read_challenge_data(current_file_name)
    
    imputer = SimpleImputer(missing_values=np.nan, fill_value=None, strategy='constant')
    imputer.fit(temp)
    
    temp = imputer.transform(temp)
    temp = preprocessing.scale(temp)
    input_training_data = np.vstack((input_training_data, temp))
    input_training_data = np.delete(input_training_data, 0, 0)  #deletes initialized array
    
    pca_initial = pca.transform(input_training_data)
    
    error = find_error(input_training_data, pca_initial, pcomponents)
    
    hours = np.hstack((pca_initial, error))
    hour_matrix = np.zeros((MAX_HOURS, TOT_FEAT))
    hour_matrix[0:len((hours)+1),:] = hours
    i = j = 0
    patient = np.zeros((MAX_HOURS, TOT_LEN))
    
    for i in range(len(hours)+1):
        row = np.zeros(TOT_LEN)
        k = 0
        
        for j in range(24):
            row[k:k + TOT_FEAT] = hour_matrix[j+i,:]

            k = ((j + 1) * TOT_FEAT)
        patient[i,:] = row


    return(patient)
    
    
#Stacks the windows and removes any extra lines
def stack_windows(output_matrix):
    DESIRED_LENGTH = 265
    stacked_matrix = np.zeros(DESIRED_LENGTH)
    patient = 0
    patient_count = len(output_matrix[0,0,:])
    for patient in range(patient_count):
        print("\r stacking patient: "+ str(patient+1) , end= '')
        for window in range(len(output_matrix)):
            test_part = output_matrix[window, -2:-13:-1, patient]
            zero = np.zeros(11)
            if(np.array_equal(test_part, zero)):
                if(window == 0):
                    stacked_matrix = np.vstack((stacked_matrix, output_matrix[0:window+1,:, patient]))
                    break
                elif(window > 0):
                    stacked_matrix = np.vstack((stacked_matrix, output_matrix[0:window,:, patient]))
                    break
                else:
                    continue  
    stacked_matrix = np.delete(stacked_matrix, 0, 0)
    return(stacked_matrix)

def stack_windows_test(output_matrix):
    DESIRED_LENGTH = 264
    stacked_matrix = np.zeros(DESIRED_LENGTH)
    window = 0
    for window in range(len(output_matrix)):
        test_part = output_matrix[window, -2:-12:-1]
        zero = np.zeros(10)
        if(np.array_equal(test_part, zero)):
            if(window == 0):
                stacked_matrix = np.vstack((stacked_matrix, output_matrix[0:window+1,:]))
                break
            else:
                stacked_matrix = np.vstack((stacked_matrix, output_matrix[0:window,:]))
                break
        else:
            continue
    stacked_matrix = np.delete(stacked_matrix, 0, 0)
    return(stacked_matrix)
#Appends the Time to Sepsis Label
def append_tts(mat_contents, patient_list, dic_label):
    tts_matrix = mat_contents[dic_label][:,np.newaxis,:]
        
    return(tts_matrix)
    
    
def scale_label(stacked_array):
    label = stacked_array[:,-1]
    for i in range(len(label)):
        if(label[i] == 999):
            label[i] = 20
        elif(label[i]<0):
            a=label[i]
            label[i]=12-a
        elif(label[i]>0):
            a=label[i]
            label[i] = 13-a
        
        

# =============================================================================
    
    #stacked_array = np.hstack((stacked_array,label))
    return(label)
                
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    