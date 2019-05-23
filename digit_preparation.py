# -*- coding: utf-8 -*-
"""
Created on Thu May 23 19:58:24 2019

@author: Asus
"""

import numpy as np 
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

DATADIR = "F:\Code\pmilu\pmlo\cleaned"
CATEGORIES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "X"]

#for category in CATEGORIES: # bacthing dataset
#    path = os.path.join(DATADIR,category) #path to digit
#    for img in os.listdir(path): # iterate over each digits per 0 to X
#        img_array = cv2.imread(os.path.join(path,img)) #convert image to array
#        plt.imshow(img_array, cmap='gray')
#        plt.show 
#        break
#    break

######################### checker ##################
training_data = []
IMG_SIZE = 50

def create_training_data():
    for category in CATEGORIES:  # batching

        path = os.path.join(DATADIR,category)  # create path to digits
        class_num = CATEGORIES.index(category)  # labelling

        for img in tqdm(os.listdir(path)):  
            try:
                img_array = cv2.imread(os.path.join(path,img))  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
         

create_training_data()

print(len(training_data))