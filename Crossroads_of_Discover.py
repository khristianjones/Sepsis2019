#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Created on Tue Oct  8 19:54:43 2019

@author: Erik
"""

import tkinter as tk
from tkinter import * 


fields = 'Age', 'Heart Rate', 'Systolic BP', 'Respiratory Rate', 'Temperature'
age =0
HR = 0 
BP = 0
RR = 0
Temp = 0

def diagnose(age, HR, BP, Temp, RR):
    age = int(age)
    if age <= 10:

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
        Temp = float(Temp)
    
        
        if RR == 0 or BP == 0 or HR == 0:
            output = "Uh oh, you're dead!"
            
        elif RR > 22 and BP < 100:
            output = "According to the QSOFA score you might have sepsis!"
        
        elif Temp > 100 and Temp < 110:
            output = "you might have a fever!"
        
        elif Temp < 97:
            output = "You better warm up!"
                    
# =============================================================================
#         elif BP > 140:
#             output = "There's a chance you have high blood pressure!"
#         
#         elif BP < 90:
#             output = "There's a chance you have low blood pressure!"
# =============================================================================

        elif RR < 25:
            output = "You have great resting heart rate!"
        
        elif RR>40:
            output ="Take a deep breath, you're breathing pretty fast!"
        else:
            output = "You seem to be doing alright"
            
    elif age > 10:
        """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        "Adults"
        
        'resting heart rate 59-99'
        'Systolic blood pressure 120-140'
        'respiratory rate 12-20'
        'Temp 98.6 deg F'
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
        Temp = float(Temp)
    
        if RR == 0 or BP == 0 or HR == 0:
            output = "Uh oh, you're dead!"
            
        elif RR > 22 and BP < 100:
            output = "According to the QSOFA score you might have sepsis!"
        
        elif Temp > 100 and Temp < 110:
            output = "you might have a fever!"
        
        elif Temp < 97:
            output = "You better warm up!"
            
        elif BP > 140:
            output = "There's a chance you have high blood pressure!"
        
        elif BP < 90:
            output = "There's a chance you have low blood pressure!"

        elif RR < 12:
            output = "You have great resting heart rate!"
        
        elif RR>20:
            output ="Take a deep breath, you're breathing pretty fast!"
        else:
            output = "You seem to be doing alright"
            
    return(output)


def select():  
    sel = "Age = " + str(age.get()) + "\n" + "Heart Rate = " + str(hr.get()) + "\n" + "Blood Pressure = " + str(bp.get()) + "\n"
    sel = sel + "Temperature = " + str(temp.get()) + "\n" + "Breathing Rate = " + str(br.get())
    biometrics = label.config(anchor=N, background = '#379683', foreground ='#EDF5E1', text = sel, justify=LEFT)  

def dispdiag():
    display = diagnose(age.get(),hr.get(),bp.get(),temp.get(),br.get())
    popup = tk.Tk()
    popup.wm_title("!")
    popup.geometry("300x50")
    popup.configure(background='#5CDB95')
    msg = display
    label = tk.Label(popup, background = '#379683', foreground ='#EDF5E1', text=msg)
    label.pack(side="top", fill="x", pady=10)
    B1 = tk.Button(popup, background = '#05386B', foreground ='#EDF5E1', text="Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()
    

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("500x500")
    root.configure(background='#5CDB95')
                  
    
    HORIZONTAL = tk.HORIZONTAL
    CENTER = tk.CENTER
    
    age = DoubleVar()
    hr = DoubleVar()
    bp = DoubleVar()
    temp = DoubleVar()
    br = DoubleVar()
    age_scale = tk.Scale(root, length = '150', width='25', background = '#379683', foreground ='#EDF5E1', troughcolor='#8EE4AF', label="Age", variable = age, from_ = 1, to = 100, orient = HORIZONTAL)  
    age_scale.pack(anchor=CENTER)  
    
    hr_scale = tk.Scale(root, length = '150', width='25', background = '#379683', foreground ='#EDF5E1', troughcolor='#8EE4AF', label = "Heart Rate",variable = hr, from_ = 0, to = 220, orient = HORIZONTAL)  
    hr_scale.pack(anchor=CENTER) 

    bp_scale = tk.Scale(root, length = '150', width='25', background = '#379683', foreground ='#EDF5E1', troughcolor='#8EE4AF', label = "Blood Pressure", variable = bp, from_ = 40, to = 200, orient = HORIZONTAL)  
    bp_scale.pack(anchor=CENTER) 

    temp_scale = tk.Scale(root, length = '150', width='25', background = '#379683', foreground ='#EDF5E1', troughcolor='#8EE4AF', label = "Temperature" ,variable = temp, from_ = 95, to = 103, resolution = 0.1, orient = HORIZONTAL)  
    temp_scale.pack(anchor=CENTER)     
  
    br_scale = tk.Scale(root, length = '150', width='25', background = '#379683', foreground ='#EDF5E1', troughcolor='#8EE4AF', label = "Breathing Rate" ,variable = br, from_ = 0, to = 50, orient = HORIZONTAL)  
    br_scale.pack(anchor=CENTER) 
    
    btn1 = tk.Button(root, background = '#05386B',text="Value", font="arial", foreground ='#EDF5E1', command=select)  
    btn1.pack(anchor=CENTER)  
    
    btn2 = tk.Button(root,background = '#05386B', text="Diagnose", font="arial", foreground ='#EDF5E1', command=dispdiag)  
    btn2.pack(anchor=CENTER)  
   
    b3 = tk.Button(root, background = '#05386B', text='Quit', font="arial", foreground ='#EDF5E1',  command=root.quit)
    b3.pack(anchor = CENTER, padx=5, pady=5)
  
    label = tk.Label(root)  
    label.pack()  
    root.mainloop()
    

    
    
    
    
    
    
    
    
    
    
    
    