import numpy as np
import cv2 
import glob as g
import os
from sklearn.model_selection import train_test_split




def get_directory():
    cur_dir = os.getcwd()


def main():
    pic = cv2.imread("metro.jpg")
    images = [cv2.imread(files) for files in g.glob("samples\*.png") ]
    #print(pic)
    #print(images)

    
    
    
    



if __name__ == "__main__":
    main()