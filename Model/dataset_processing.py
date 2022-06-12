import cv2 
import numpy as np
import glob
import os

main_dir = "F:\Capstone Project\Model"


def extractSkin(img):
    

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    

    new_img = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(new_img, 80, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return thresh1

def populate_dir(name):
    cur_dir = os.getcwd()
    dataset_dir = "F:\Capstone Project\\Own_Dataset"
    dataset_dir = dataset_dir + "\\" + "Dataset" + name + "\\" + "*.jpg"
    #print(dataset_dir)
    dest_path = cur_dir + "\\" + name 
    #images = [cv2.imread(files) for files in glob.glob(dataset_dir) ]
    images = glob.glob(dataset_dir)
    #print(images)
    os.chdir(dest_path)
    i = 0
    print("Entering For Loop", name )
    for sample in images:
        i += 1
        pivot_img = cv2.imread(sample, 1)
        #pivot_img = cv2.cvtColor(pivot_img, cv2.COLOR_BGR2RGB)
        
        thresh1 = extractSkin(pivot_img)
        #resized_image = cv2.resize( pivot_img, (224, 224) )

        #ret, thresh1 = cv2.threshold(resized_image, 120, 255, cv2.THRESH_BINARY)
        #blur = cv2.GaussianBlur(resized_image, (5,5), 0)
        #ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        image_name = "sample_" + str(i) + ".jpg"
        cv2.imwrite(image_name, thresh1)
        print("writing image - ", i )


    print("Finished", name)
    os.chdir(main_dir)



def main():
    sign = 10
    
    for i in range(1, sign + 1):
        cur_dir = os.getcwd()
        cur_dir = cur_dir + "\\" +str(i) 
        if not os.path.exists(cur_dir):
            os.makedirs( str(i) )
        populate_dir( str(i) )






if __name__ == "__main__":
    main()