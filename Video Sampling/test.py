import cv2
import os


def getframes(cur_dir, video, sec, count):
    video.set(cv2.CAP_PROP_POS_MSEC, sec*1000)
    hasframes, images = video.read()
    print(count)
    if hasframes:
        
        cv2.imwrite(cur_dir+"\image" + str(count) + ".jpg", images )
    return hasframes

def main():
    cur_dir = os.getcwd()
    cur_dir = cur_dir + "\samples"

    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)
        
    video = cv2.VideoCapture("clipped.mp4")
    sec = 0
    frameRate = 0.1
    count = 1
    success = getframes(cur_dir, video, sec, count)
    print(count)
    
    while (success):
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getframes(cur_dir, video, sec, count)
    
    print("Successful!!!")
        




if __name__ == "__main__":
    main()