#new version as of June 10 2019
from sklearn.decomposition import PCA
from tensorflow import keras
import numpy as np
import functions

def load_sepsis_model():
    stacked_train1 = np.load("stacked_train1.npy")
    stacked_train2 = np.load("stacked_train2.npy")
    stacked_train3 = np.load("stacked_train3.npy")
    stacked_train4 = np.load("stacked_train4.npy")
    stacked_train = np.vstack((stacked_train1, stacked_train2))
    stacked_train = np.vstack((stacked_train, stacked_train3))
    stacked_train = np.vstack((stacked_train, stacked_train3))
    stacked_train = np.vstack((stacked_train, stacked_train4))
    
    model1 = keras.models.load_model('my_model.h5')
    model = (model1, stacked_train)
    return(model)
    
def get_sepsis_score(current_data, model):

    model1 = model[0]
    stacked_train = model[1]

    pca = PCA(n_components=10)
    pca.fit(stacked_train)
    
    test_patient = functions.hour_by_hour(current_data)

    #PCA requires a 2D array. This bit ensures that if it is the first hour, then the patient will have 2D
    if test_patient.size == 40:
        test_patient = np.vstack((test_patient, test_patient))
        pca_test = pca.transform(test_patient)
    else:
        pca_test = pca.transform(test_patient)
        
    output=model1.predict(pca_test[:,:])
    
    if output[-1,1] >= .1:
        current_score = output[-1,1] * 10
        current_label = 1
    else:
        current_score = output[-1,1] * 10
        current_label = 0
        
    return(current_score, current_label)

