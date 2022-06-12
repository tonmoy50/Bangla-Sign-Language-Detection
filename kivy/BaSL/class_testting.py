import detect_sign
from detect_sign import SignDetection


sd = SignDetection(image_path='sample_1.jpg')
print(sd.class_name)