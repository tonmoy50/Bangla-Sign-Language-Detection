import numpy as np
from keras.preprocessing.image import ImageDataGenerator
#from keras.applications import VGG16
import imageio
import cv2
#import  torchvision.transforms.functional as TF
from imgaug import augmenters as iaa
import glob

path = "F:\Capstone Project\Model"

def main():

    direct = "F:\Capstone Project\Model\\train_mega\\1"

    img_set = glob.glob(direct)
    #print(img_set)

    for samples in img_set:
        i = cv2.imread(samples, 1)
        print(i)

    img = imageio.imread("img1.jpg")
    rotate = iaa.Affine(rotate=(-25, 25))
    image_rotated = rotate.augment_images([img])[0]
    cv2.imwrite("aug.jpg", image_rotated)







if __name__ == "__main__":
    main()