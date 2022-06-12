import cv2
import numpy as np 


def main():
    pic = cv2.imread("temp.jpg")

    cv2.imshow("Original", pic)

    scale = 50

    #width = int( pic.shape[1]*scale / 100 )
    #height = int( pic.shape[0]*scale / 100 )
    width = 320
    height = 240
    
    dim = (width, height)

    downscaled = cv2.resize( pic, dim, interpolation= cv2.INTER_AREA )
    grayed = cv2.cvtColor(downscaled, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("bnw.jpg", grayed)

    ret, threshed = cv2.threshold(grayed, 50, 255, cv2.THRESH_BINARY)
    mean = cv2.adaptiveThreshold(grayed,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
    gaussian = cv2.adaptiveThreshold(grayed,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    cv2.imwrite("downscaled.jpg", downscaled)
    cv2.imwrite("threshed.jpg", threshed)
    cv2.imwrite("meand.jpg", mean)
    cv2.imwrite("gauss.jpg", gaussian)


    


if __name__ == "__main__":
    main()

