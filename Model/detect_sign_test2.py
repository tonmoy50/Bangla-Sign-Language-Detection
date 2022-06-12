import keras
from keras.models import load_model
import cv2
import glob
import numpy as np
import pandas as pd




def get_file(filepath):
    data = pd.read_csv(filepath)
    data = np.array(data)
    return data

path = "F:\Capstone Project\Model\database\word.csv"
data = get_file(path)

model = load_model('now.h5')

model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

img = [cv2.imread(files) for files in glob.glob("F:\Capstone Project\Model\\train2\\10\*.jpg") ]

class_list = []
for sample in img:

    sample = cv2.resize(sample,(224,224))

    sample = np.reshape(sample,[1,224,224,3])

    classes = model.predict_classes(sample)
    class_list.append(classes)
    #print(data[classes])
    print(classes)
print(data)
