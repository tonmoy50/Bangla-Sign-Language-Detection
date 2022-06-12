import numpy as np
import cv2



path = "F:\Capstone Project\Model\\bulk\\test.jpg"

img = cv2.imread(path, 1)
img = cv2.resize(img, (600,800))

img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
upper_threshold = np.array([20, 255, 255], dtype=np.uint8)
skinMask = cv2.inRange(img, lower_threshold, upper_threshold)
skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
skin = cv2.bitwise_and(img, img, mask=skinMask)
new_img = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)


cv2.imshow("Image", new_img)


cv2.waitKey(0)

cv2.destroyAllWindows()