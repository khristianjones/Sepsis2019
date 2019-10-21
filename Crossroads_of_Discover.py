#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Created on Tue Oct  8 19:54:43 2019

@author: Erik
"""
# =============================================================================
# from tkinter import *
# import tkinter as tk
# root = tk.Tk()
# 
# Label(root, text = "Childs First name").grid(row = 0, column = 0, height = 5, width = 10, sticky = W)
# Label(root, text = "Childs Surname").grid(row = 1, column = 0, height = 5, width = 10, sticky = W)
# Label(root, text = "Childs Year of Birth").grid(row = 2, column = 0, height = 5, width = 10, sticky = W)
# Label(root, text = "Childs Month of Birth").grid(row = 3, column = 1, height = 5, width = 10, sticky = W)
# Label(root, text = "Childs Day of Birth").grid(row = 4, column = 1, height = 5, width = 10, sticky = W)
# 
# Fname = Entry(root)
# Sname = Entry(root)
# x = Entry(root)
# y = Entry(root)
# z = Entry(root)
# 
# 
# Fname.grid(row = 0, column = 1)
# Sname.grid(row = 1, column = 1)
# x.grid(row = 3, column = 1)
# y.grid(row = 2, column = 1)
# z.grid(row = 4, column = 1)
# 
# def getInput():
# 
#     a = Fname.get()
#     b = Sname.get()
#     c = x.get()
#     d = y.get()
#     e = z.get()
#     root.destroy()
# 
#     global params
#     params = [a,b,c,d,e]
# 
# 
# button = Button(root, text = "submit",
#            command = getInput).grid(row = 5, sticky = W)
# 
# root.mainloop()
# =============================================================================
import tkinter as tk
global age, HR, BP, RR, Temp
fields = 'Age', 'Heart Rate', 'Systolic BP', 'Respiratory Rate', 'Temperature'
age =0
HR = 0 
BP = 0
RR = 0
Temp = 0
def display(entries):
    for entry in entries:
        field = entry[0]
        text  = entry[1].get()
        print('%s: "%s"' % (field, text)) 
def fetch(entries):
    global age, HR, BP, RR, Temp
    age = entries[0][1]
    HR = entries[1][1] 
    BP = entries[2][1]
    RR = entries[3][1]
    Temp = entries[4][1]
       

def makeform(root, fields):
    entries = []
    for field in fields:
        row = tk.Frame(root)
        lab = tk.Label(row, width=15, text=field, anchor='w')
        ent = tk.Entry(row)
        row.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        lab.pack(side=tk.LEFT)
        ent.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        entries.append((field, ent))
    return entries

def popupmsg():
    popup = tk.Tk()
    popup.wm_title("!")
    msg = diagnose()
    label = tk.Label(popup, text=msg)
    label.pack(side="top", fill="x", pady=10)
    B1 = tk.Button(popup, text="Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()
    
def diagnose():
    global age, HR, BP, RR, Temp
    age = int(age)
    if age < 10:
        print('made it')
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        "Children 0-10"
        
        'resting heart rate 69-129'
        'Systolic blood pressure 80-120'
        'Respiratpory rate 18-34'
        'Temp 98'
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    
        'Enter a value for HR in BPM'
        HR = int(HR)
        'Enter a value for systolic BP in mmHG'
        #BP = input("Enter your Systolic Blood Pressure in mmHg: ")
        BP =int(BP)
        'Enter a value for RR in BPM'
        #RR = input("Enter your respiratory Rate in BPM: ")
        RR = int(RR)
        'Enter a value for Temp in deg F'
        #Temp = input("Enter your temperature in Fahrenheit: ")
        Temp = int(Temp)
    
        
        if RR == 0 or BP == 0 or HR == 0:
            output = "He's dead Jim!"
            
        elif RR > 22 and BP < 100:
            output = "you might have sepsis!"
        
        elif Temp > 100 and Temp < 110:
            output = "you might have a fever!"
        
        elif Temp < 97 and Temp < 50:
            output = "You better warm up!"
    return(output)
# =============================================================================
# elif age > 10:
#     """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#     "Adults"
#     
#     'resting heart rate 59-99'
#     'Systolic blood pressure 120-140'
#     'respiratory rate 12-20'
#     'Temp 98.6 deg F'
#     """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# 
#     'Enter a value for HR in BPM'
#     HR = input("Enter your Heart Rate in BPM: ")
#     HR = int(HR)
#     'Enter a value for systolic BP in mmHG'
#     BP = input("Enter your Systolic Blood Pressure in mmHg: ")
#     BP =int(BP)
#     'Enter a value for RR in BPM'
#     RR = input("Enter your respiratory Rate in BPM: ")
#     RR = int(RR)
#     'Enter a value for Temp in deg F'
#     Temp = input("Enter your temperature in Fahrenheit: ")
#     Temp = int(Temp)
# 
#     
#     if RR == 0 or BP == 0 or HR == 0:
#         print("He's dead Jim!")
#         
#     elif RR > 22 and BP < 100:
#         print("you might have sepsis!")
#     
#     elif Temp > 100 and Temp < 110:
#         print("you might have a fever!")
#     
#     elif Temp < 97 and Temp < 50:
#         print("You better warm up!")
# =============================================================================

if __name__ == '__main__':
    root = tk.Tk()
    ents = makeform(root, fields)
    #root.bind('', (lambda event, e=ents: fetch(e)))   
    #original#b1 = tk.Button(root, text='Diagnose',
    #              command=(lambda e=ents: fetch(e)))
    b1 = tk.Button(root, text = 'Enter',
                  command=(lambda e=ents: fetch(e)))
    b1.pack(side=tk.LEFT, padx=5, pady=5)
    b2 = tk.Button(root, text = 'Diagnose',
                  command=popupmsg)
    b2.pack(side=tk.LEFT, padx=5, pady=5)
    b3 = tk.Button(root, text='Quit', command=root.quit)
    b3.pack(side=tk.LEFT, padx=5, pady=5)
    root.mainloop()
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    