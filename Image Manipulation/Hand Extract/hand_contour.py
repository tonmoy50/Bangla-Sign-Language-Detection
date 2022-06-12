import cv2 
import numpy as np


def main():
    img = cv2.imread('processed1.jpg', 0)
    #img = cv2.resize(img, (128, 128))

    ret, thresh1 = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    countour, x = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    hull = [cv2.convexHull(c) for c in countour]
    final = cv2.drawContours(img, hull, -1, (255,0,0) )

    print(len(countour))

    for i in range(len(countour)):
        mat = img[countour[i]]
    


    #cv2.imshow('temp', mat)
    cv2.imshow('greyscaled',img)
    #cv2.imshow('Threshhold',thresh1)
    #cv2.imshow('Contour',final)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    #cv2.imwrite("dfsd.jpg", thresh1)




if __name__ == "__main__":
    main()