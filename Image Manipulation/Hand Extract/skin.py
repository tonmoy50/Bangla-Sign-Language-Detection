import numpy as np
import cv2


def extractSkin(img):
    
    imag = cv2.resize(img, (400,400))
    cv2.imshow("raw",imag)
    

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    

    new_img = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(new_img, 80, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    new_img = cv2.resize(new_img, (400,400))
    thresh1 = cv2.resize(thresh1, (400,400))
    cv2.imshow("skin",new_img)
    cv2.imshow("threshed", thresh1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



def test(img):
    
    img = cv2.resize(img, (316,316))
    skin = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    skin = cv2.GaussianBlur(skin, (3,3), 0)
    skin = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    cv2.imshow("HSV",skin)

    ret, thresh1 = cv2.threshold(skin, 240, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh2 = cv2.adaptiveThreshold(skin, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh3 = cv2.adaptiveThreshold(skin, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh4 = cv2.adaptiveThreshold(skin, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh5 = cv2.adaptiveThreshold(skin, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)

    skinMask = cv2.inRange(skin, lower_threshold, upper_threshold)

    skin = cv2.bitwise_and(img, img, mask=skinMask)
    
    
    #cv2.imshow("skinMask",skinMask)
    cv2.imshow("thresh1",thresh1)
    cv2.imshow("thresh2",thresh2)
    cv2.imshow("thresh3",thresh3)
    cv2.imshow("thresh4",thresh4)
    cv2.imshow("thresh5",thresh5)
    
    


    cv2.waitKey(0)
    cv2.destroyAllWindows()



def main():
    img = cv2.imread("af.jpg", 1)
    
    #test(img)
    extractSkin(img)

if __name__ == "__main__":
    main()