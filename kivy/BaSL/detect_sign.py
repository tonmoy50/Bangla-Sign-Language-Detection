import keras
from keras.models import load_model
import cv2
import glob
import numpy as np
import pandas as pd
import os
from t_to_s import *

class SignDetection():
    model_name = 'test_model1.h5'
    

    def __init__(self, image_path):
        self.dataset_name_path = os.path.join(os.getcwd(), 'database\word.csv')    #"F:\Capstone Project\Model\database\word.csv"
        self.model = load_model(self.model_name)
        self.image_path = image_path
        self.class_name = self.detect()
        

    def get_file(self, filepath):
        data = pd.read_csv(filepath)
        data = np.array(data)
        return data 

    def detect(self):

        self.model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        img = cv2.imread(self.image_path)

        img_height, img_width = 128, 128

        sample = cv2.resize(img, (img_height, img_width) )
        sample = np.reshape(sample, [1, img_height, img_width, 3])
        
        class_val = self.model.predict_classes(sample)
        data = self.get_file(self.dataset_name_path)

        

        if class_val == 0:
            class_val = 1
            formatted_data = data[class_val]
            
            formatted_data = formatted_data[0]
        elif class_val == 1:
            class_val = 10
            formatted_data = data[class_val]
            
            formatted_data = formatted_data[0]
        else:
            class_val = class_val + 1
            formatted_data = data[class_val]
            formatted_data = formatted_data[0]
            formatted_data = formatted_data[0]
            

        
        #formatted_data = formatted_data[0]
        #formatted_data = formatted_data[0]

        #self.translate()

        return formatted_data


