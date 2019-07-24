#new version as of June 10 2019
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import warnings
warnings.filterwarnings('ignore') #Removes warnings from preprocessing scale
PATH = "C:\\Users\\Erik.IPDPC-63\\Desktop\\Whitaker Research\\featureLearner\\training\\p00001.psv"
PATH2 = "C:\\Users\\Erik.IPDPC-63\\Desktop\\Whitaker Research\\featureLearner\\training\\p00800.psv"
TESTPATH = "C:\\Users\\Erik.IPDPC-63\\Desktop\\Whitaker Research\\featureLearner\\training\\p00013.psv"
# =============================================================================
# file = open(PATH, 'r')
# for line in file:
#     print(line + '\n')
# 
# file.close()
# =============================================================================

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


non_sepsis_count = 1 
sepsis_count = 0
non_sepsis_list = ['training\\p00001.psv'] 
sepsis_list = []
########################Sepsis patients = 58 | Non Sepsis patients = 4942######
for i in range(50): #4999
    i+=2
    current_file = 'training\\p{0:05d}.psv'.format(i)
    patient, column_names = read_challenge_data(current_file)
    for l in range(len(patient)):
        x = patient[l,40]
        print(x)
        if x == 1:
            sepsis_count +=1
            sepsis_list.append(current_file)
            break
        else:
            non_sepsis_count +=1
            non_sepsis_list.append(current_file)
            break
print('number of non-sepsis patients', non_sepsis_count)
print('number of sepsis patients', sepsis_count)
test, col = read_challenge_data(TESTPATH)
display(test)
print(test.shape)
print

#################Splits Patients into training and test and normalizes#########
data1, column = (read_challenge_data(PATH))
imp_frequent = SimpleImputer(missing_values=np.nan, fill_value=None, strategy='constant')
data1 = imp_frequent.fit_transform(data1)
data1 = preprocessing.scale(data1)


###############Draft for training set##########################################
for i in range(4000):    
    i+=2
    current_file = 'training\\p{0:05d}.psv'.format(i)
    temp1, column_names1 = read_challenge_data(current_file)
   
    imp_frequent = SimpleImputer(missing_values=np.nan, fill_value=None, strategy='constant')
    imp_frequent.fit(temp1)
    train_filled = imp_frequent.transform(temp1)
    train_scaled = preprocessing.scale(train_filled)
    
    data1 = np.vstack((data1, train_scaled))

start2, column  = read_challenge_data(PATH2)
imp_frequent = SimpleImputer(missing_values=np.nan, fill_value=None, strategy='constant')
start2 = imp_frequent.fit_transform(start2)
start2 = preprocessing.scale(start2)

###################Draft for test Set##########################################
for l in range(4000, 5000):    #Stacks patients on top of each other
    current_file = 'training\\p{0:05d}.psv'.format(l)
    temp2, column_names = read_challenge_data(current_file)
   
    imp_frequent = SimpleImputer(missing_values=np.nan, fill_value=None, strategy='constant')
    imp_frequent.fit(temp2)
    test_filled = imp_frequent.transform(temp2)
    test_scaled = preprocessing.scale(test_filled)

    
    test_data = np.vstack((data1, test_scaled))
    
pca = PCA(n_components = 3)

pca.fit(data1)

data1 = pca.transform(data1)
test_data = pca.transform(test_data)




##############Getting sepsis patients##########################################

start3, column  = read_challenge_data(sepsis_list[0])
imp_frequent = SimpleImputer(missing_values=np.nan, fill_value=None, strategy='constant')
start3 = imp_frequent.fit_transform(start3)
start3 = preprocessing.scale(start3)

for k in range(len(sepsis_list)-1):
    k+=1
    temp3, column_names = read_challenge_data(sepsis_list[k])
   
    imp_frequent = SimpleImputer(missing_values=np.nan, fill_value=None, strategy='constant')
    imp_frequent.fit(temp3)
    sep_filled = imp_frequent.transform(temp3)
    sep_scaled = preprocessing.scale(sep_filled)

    
    sepsis = np.vstack((start3, sep_scaled))
print('# of sepsis', k)
sepsis = pca.transform(sepsis)
##############KMeans Clustering################################################

kmeans = KMeans(n_clusters = 7)

kmeans = kmeans.fit(data1)

labels = kmeans.predict(data1)

centroids = kmeans.cluster_centers_
colors = ['r', 'g', 'y']
fig = plt.figure()

ax = Axes3D(fig)
ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], c=labels, cmap = 'viridis', s = 1 ) #, data1[:, 2]
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='*', s = 5, color = 'b') #, centroids[:, 2]
ax.scatter(sepsis[:, 0], sepsis[:, 1], sepsis[:, 2], c = 'black', s = 10)

fig2 = plt.figure()

plt.scatter(data1[:, 0], data1[:, 1], s=5, c=labels, cmap = 'viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s = 5, color = 'b')
plt.scatter(sepsis[:, 0], sepsis[:, 1], color = 'black', s = 10)

fig3 = plt.figure()

plt.scatter(data1[:, 0], data1[:, 2], s=5, c=labels, cmap = 'viridis')
plt.scatter(centroids[:, 0], centroids[:, 2], marker='*', s = 5, color = 'b')
plt.scatter(sepsis[:, 0], sepsis[:, 2], color = 'black', s = 10)

fig4 = plt.figure()

plt.scatter(data1[:, 1], data1[:, 2], s=5, c=labels, cmap = 'viridis')
plt.scatter(centroids[:, 1], centroids[:, 2], marker='*', s = 5, color = 'b')
plt.scatter(sepsis[:, 1], sepsis[:, 2], color = 'black', s = 10)

