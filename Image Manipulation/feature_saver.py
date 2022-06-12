import numpy as np
import pandas as pd
import cv2 
from keras.preprocessing.image import image




def main():
    img = cv2.imread('tito.jpg', 0)
    
    img = img.flatten()
    print(img)
    np.save('test.npy', img)

if __name__ == "__main__":
    main()