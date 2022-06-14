from __future__ import print_function

import matplotlib.pyplot as plt

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# from keras.applications import VGG16
import imageio
import cv2
import glob

# import  torchvision.transforms.functional as TF
from imgaug import augmenters as iaa
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from skimage.io import imread, imsave
from skimage import exposure, color
from skimage.transform import resize
import os


path = "F:\Capstone Project\Model"
save_location = "F:\Capstone Project\Model\\2"


def imgGen(
    batch_num,
    img,
    zca=False,
    rotation=0.0,
    w_shift=0.0,
    h_shift=0.0,
    shear=0.0,
    zoom=0.3,
    h_flip=False,
    v_flip=False,
    preprocess_fcn=None,
    batch_size=9,
):
    datagen = ImageDataGenerator(
        zca_whitening=zca,
        rotation_range=rotation,
        width_shift_range=w_shift,
        height_shift_range=h_shift,
        shear_range=shear,
        zoom_range=zoom,
        fill_mode="nearest",
        horizontal_flip=h_flip,
        vertical_flip=v_flip,
        preprocessing_function=preprocess_fcn,
        data_format=K.image_data_format(),
    )

    datagen.fit(img)

    i = 0
    img_num = batch_num
    for img_batch in datagen.flow(img, batch_size=9, shuffle=False):
        for img in img_batch:
            print("Working on Image {}".format(img_num))
            modder = "\\" + str(img_num) + ".jpg"
            img_num += 1

            img = 255 * img
            img = img.astype(np.uint8)

            if not os.path.exists(save_location):
                os.mkdir(save_location)
            
            imsave(save_location + modder, img)
            i = i + 1
        if i >= batch_size:
            break
    # plt.show()


def main():

    start = 1
    end = 7

    while start <= end:
        # direct = "F:\Capstone Project\sobar_pic\Dataset" + str(start) + "\*.jpg"

        direct = os.path.join(os.getcwd(), "Bangla Sign Language Dataset", str(start), "*.jpg")


        global save_location
        save_location = "F:\Capstone Project\Model\\" + str(start)
        image_list = glob.glob(direct)
        print("Working on Image batch {}".format(start))
        batch_number = 1
        for sample in image_list:
            img = imread(sample)
            img = img.astype("float32")
            img /= 255
            h_dim = np.shape(img)[0]
            w_dim = np.shape(img)[1]
            num_channel = np.shape(img)[2]
            img = img.reshape(1, h_dim, w_dim, num_channel)
            # print(img.shape)
            imgGen(batch_number, img, rotation=15, h_shift=0.5)
            batch_number += 9

        start += 1

    """
    direct = "F:\Capstone Project\Own_Dataset\Dataset2\*.jpg"
    image_list = glob.glob(direct)
    batch_number = 1
    for sample in image_list:
        img = imread(sample)
        img = img.astype('float32')
        img /= 255
        h_dim = np.shape(img)[0]
        w_dim = np.shape(img)[1]
        num_channel = np.shape(img)[2]
        img = img.reshape(1, h_dim, w_dim, num_channel)
        #print(img.shape)
        imgGen(batch_number, img, rotation=15, h_shift=0.5)
        batch_number += 9
    """

    """
    img_path = "F:\Capstone Project\Model\\bulk\\1.jpg"
    img = imread(img_path)
    #plt.imshow(img)
    #plt.show()

    img = img.astype('float32')
    img /= 255
    h_dim = np.shape(img)[0]
    w_dim = np.shape(img)[1]
    num_channel = np.shape(img)[2]
    img = img.reshape(1, h_dim, w_dim, num_channel)
    print(img.shape)


    imgGen(img, rotation=30, h_shift=0.5)
    """


if __name__ == "__main__":
    main()
