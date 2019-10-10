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
import matplotlib
matplotlib.rcParams["backend"] = "TkAgg"
import matplotlib.pyplot as plt

stacked_train1 = np.load("stacked_train1.npy")
stacked_train2 = np.load("stacked_train2.npy")
stacked_train3 = np.load("stacked_train3.npy")
stacked_train4 = np.load("stacked_train4.npy")
stacked_train = np.vstack((stacked_train1, stacked_train2))
stacked_train = np.vstack((stacked_train, stacked_train3))
stacked_train = np.vstack((stacked_train, stacked_train3))
stacked_train = np.vstack((stacked_train, stacked_train4))
    
pca = PCA(n_components=40)
pca.fit(stacked_train)

evr = np.zeros(40)

evr = pca.explained_variance_ratio_

tot_evr = np.zeros(40)
for i in range(40):
    tot_evr[i] = sum(evr[0:i])

pca2 = PCA(n_components=10)
pca2.fit(stacked_train)

pca3 = PCA(n_components=1)
pca3.fit(stacked_train)

print((pca2.components_).shape)

pcomp = np.zeros([40,40])

pcomp = pca.components_

pcomp2 = np.zeros([10,40])

pcomp2 = pca2.components_

pcomp3 = np.zeros(40)

pcomp3 = pca3.components_

xpos = np.arange(40)
xpos = xpos+1




evr = evr.tolist()
tot_evr = tot_evr.tolist()
xpos2 = [1, 5, 10, 15, 20, 25, 30, 35, 40]  
xpos = xpos.tolist()
plt.bar(xpos, tot_evr, 0.75, align='center')

plt.title('Explained Variance Ratio')
plt.ylabel('Explained Variance')
plt.xlabel('Feature')
plt.xticks(xpos2, xpos2, rotation = 0)
plt.axis([0.5, 40.5, 0, 1], 'scaled')
plt.annotate('Chosen # of Components',
             xy=(10,.78),
             xytext=(0,1))
             

plt.savefig('explained_variance_graph')
# =============================================================================
# sepsis = np.array([73, 27])
# non_sepsis = np.array([91, 9])
# xpos = np.array([0, 1])
# sep_objects = ('True\nPositive', 'False \nPositive')
# non_objects = ('True \nNegative', 'False \nNegative')
# 
# plt.bar(xpos, sepsis, 0.5, align='center', alpha=0.5)
# 
# plt.title('Sepsis Patient Prediction Accuracy')
# plt.ylabel('percent ')
# plt.xticks(xpos, sep_objects)
# plt.axis([-0.5, 1.5, 0, 100], 'scaled')  
# plt.show()
# 
# plt.bar(xpos, non_sepsis, 0.5, align='center', alpha=0.5)
# 
# plt.title('Non-Sepsis Patient Prediction Accuracy')
# plt.ylabel('percent ')
# plt.xticks(xpos, non_objects)
# plt.axis([-0.5, 1.5, 0, 100], 'scaled')  
# plt.show()
# 
# import matplotlib
# # Specifying the backend to be used before importing pyplot
# # to avoid "RuntimeError: Invalid DISPLAY variable"
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
# import numpy as np
# 
# class TrainingPlot(keras.callbacks.Callback):
# 
#     # This function is called when the training begins
#     def on_train_begin(self, logs={}):
#         # Initialize the lists for holding the logs, losses and accuracies
#         self.losses = []
#         self.acc = []
#         self.val_losses = []
#         self.val_acc = []
#         self.logs = []
# 
#     # This function is called at the end of each epoch
#     def on_epoch_end(self, epoch, logs={}):
# 
#         # Append the logs, losses and accuracies to the lists
#         self.logs.append(logs)
#         self.losses.append(logs.get('loss'))
#         self.acc.append(logs.get('acc'))
#         self.val_losses.append(logs.get('val_loss'))
#         self.val_acc.append(logs.get('val_acc'))
# 
#         # Before plotting ensure at least 2 epochs have passed
#         if len(self.losses) > 1:
# 
#             N = np.arange(0, len(self.losses))
# 
#             # You can chose the style of your preference
#             # print(plt.style.available) to see the available options
#             #plt.style.use("seaborn")
# 
#             # Plot train loss, train acc, val loss and val acc against epochs passed
#             plt.figure()
#             plt.plot(N, self.losses, label = "train_loss")
#             plt.plot(N, self.acc, label = "train_acc")
#             plt.plot(N, self.val_losses, label = "val_loss")
#             plt.plot(N, self.val_acc, label = "val_acc")
#             plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
#             plt.xlabel("Epoch #")
#             plt.ylabel("Loss/Accuracy")
#             plt.legend()
#             # Make sure there exists a folder called output in the current directory
#             # or replace 'output' with whatever direcory you want to put in the plots
#             plt.savefig('figures/Epoch-{}.png'.format(epoch))
#             plt.close()
# 
# =============================================================================
