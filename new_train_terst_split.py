"""
Created on Fri Aug  2 14:10:19 2019

@author: Erik
"""
import os
import os.path
import random

#filename = os.path.join(dir, 'relative','path','to','file','you','want')

total_list1 = []

#####################Creates train test for the first file, I believe this only works on windows###################
for filename in os.listdir("Training2/trainingA/training/"):
    x = filename[1:7]
    x = x.lstrip("0")   #Strips Leading zeros
    total_list1.append(x)

random.shuffle(total_list1)

x = len(total_list1)

twenty = int(x*0.2)

eighty = int(x*0.8)
print(twenty, eighty)

train_setA = total_list1[0: eighty]
test_setA = total_list1[eighty: x]

f=open("train_setA.txt", "a")
i=0
for i in range(len(test_setA)):
    
    f.write(test_setA[i])
    f.write("\t")
    i+=1
f.close()

f=open("test_setA.txt", "a")
i=0
for i in range(len(test_setA)):
    
    f.write(test_setA[i])
    f.write("\t")
    i+=1
f.close()


#####################Creates train test for the second file###################
total_list2 = []
for filename in os.listdir("Training2/trainingB/training_setB/"):
    x = filename[1:7]
    total_list2.append(x)

random.shuffle(total_list2)

x = len(total_list2)

twenty = int(x*0.2)

eighty = int(x*0.8)
print(twenty, eighty)

train_setB = total_list2[0: eighty]
test_setB = total_list2[eighty: x]

f=open("train_setB.txt", "a")
i=0
for i in range(len(train_setB)):
    
    f.write(train_setB[i])
    f.write("\t")
    i+=1
f.close()

f=open("test_setB.txt", "a")
i=0
for i in range(len(test_setB)):
    
    f.write(test_setB[i])
    f.write("\t")
    i+=1
f.close()





