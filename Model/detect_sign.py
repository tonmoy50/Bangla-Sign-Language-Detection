import keras
from keras.models import load_model
import cv2
import glob
import numpy as np
 

model = load_model('model_1.h5')

model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

img = [cv2.imread(files) for files in glob.glob("bulk\*jpg") ]

class_list = []
for sample in img:

    sample = cv2.resize(sample,(128,128))

    sample = np.reshape(sample,[1,128,128,3])

    classes = model.predict_classes(sample)
    class_list.append(classes)
    print(classes)