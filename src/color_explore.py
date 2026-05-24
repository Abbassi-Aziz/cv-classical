import cv2
import numpy as np

img = np.zeros((512,512,3), dtype=np.uint8)
red = img.copy()
red[:,:,2] = 255
green = img.copy()
green[:,:,1] = 255
blue = img.copy()
blue[:,:,0] = 255

red_gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)
green_gray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
blue_gray = cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)

print("red_gray_center: ",red_gray[256,256])
print("green_gray_center: ",green_gray[256,256])
print("blue_gray_center: ",blue_gray[256,256])