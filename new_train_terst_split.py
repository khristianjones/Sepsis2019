"""
Created on Fri Aug  2 14:10:19 2019

@author: Erik
"""
import os
import os.path
import random

#filename = os.path.join(dir, 'relative','path','to','file','you','want')

total_list = []
for filename in os.listdir("Training2/trainingA/training/"):
    x = filename[1:7]
    x = x.lstrip("0")   #Strips Leading zeros
    total_list.append(x)

random.shuffle(total_list)

x = len(total_list)

twenty = int(x*0.2)

eighty = int(x*0.8)
print(twenty, eighty)

train = total_list[0: eighty]
test = total_list[eighty: x]

f=open("new_test.txt", "a")
i=0
for i in range(len(test)):
    
    f.write(test[i])
    f.write("\t")
    i+=1
f.close()










