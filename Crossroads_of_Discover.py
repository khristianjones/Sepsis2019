#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Created on Tue Oct  8 19:54:43 2019

@author: Erik
"""
from tkinter import *

window = Tk()

window = title("Welcome")

window.mainloop()

age = input("Enter your Age: ")
age = int(age)
if age < 10:
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    "Children 0-10"
    
    'resting heart rate 69-129'
    'Systolic blood pressure 80-120'
    'Respiratpory rate 18-34'
    'Temp 98'
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    'Enter a value for HR in BPM'
    HR = input("Enter your Heart Rate in BPM: ")
    HR = int(HR)
    'Enter a value for systolic BP in mmHG'
    BP = input("Enter your Systolic Blood Pressure in mmHg: ")
    BP =int(BP)
    'Enter a value for RR in BPM'
    RR = input("Enter your respiratory Rate in BPM: ")
    RR = int(RR)
    'Enter a value for Temp in deg F'
    Temp = input("Enter your temperature in Fahrenheit: ")
    Temp = int(Temp)

    
    if RR == 0 or BP == 0 or HR == 0:
        print("He's dead Jim!")
        
    elif RR > 22 and BP < 100:
        print("you might have sepsis!")
    
    elif Temp > 100 and Temp < 110:
        print("you might have a fever!")
    
    elif Temp < 97 and Temp < 50:
        print("You better warm up!")

elif age > 10:
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    "Adults"
    
    'resting heart rate 59-99'
    'Systolic blood pressure 120-140'
    'respiratory rate 12-20'
    'Temp 98.6 deg F'
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    'Enter a value for HR in BPM'
    HR = input("Enter your Heart Rate in BPM: ")
    HR = int(HR)
    'Enter a value for systolic BP in mmHG'
    BP = input("Enter your Systolic Blood Pressure in mmHg: ")
    BP =int(BP)
    'Enter a value for RR in BPM'
    RR = input("Enter your respiratory Rate in BPM: ")
    RR = int(RR)
    'Enter a value for Temp in deg F'
    Temp = input("Enter your temperature in Fahrenheit: ")
    Temp = int(Temp)

    
    if RR == 0 or BP == 0 or HR == 0:
        print("He's dead Jim!")
        
    elif RR > 22 and BP < 100:
        print("you might have sepsis!")
    
    elif Temp > 100 and Temp < 110:
        print("you might have a fever!")
    
    elif Temp < 97 and Temp < 50:
        print("You better warm up!")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    