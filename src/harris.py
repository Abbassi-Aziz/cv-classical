import cv2 
import numpy as np
import os
from cvlib.utils import get_project_paths
from cvlib.filters import conv_2d

paths = get_project_paths(__file__)
img_path = os.path.join(paths["data"],"butterfly.jpg")

img = cv2.imread(img_path)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray,(5,5),1.0)

Gx = np.array([[-1,0,1],
               [-2,0,2],
               [-1,0,1]], dtype=np.float32)
Gy = np.array([[-1,-2,-1],
               [0,0,0],
               [1,2,1]], dtype=np.float32)

Ix = conv_2d(blurred, Gx)
Iy = conv_2d(blurred,Gy)

Ix2 = Ix * Ix
Iy2 = Iy * Iy
Ixy = Ix * Iy

Ix2 = cv2.GaussianBlur(Ix2,(5,5),1.0,cv2.CV_32F)
Iy2 = cv2.GaussianBlur(Iy2,(5,5),1.0,cv2.CV_32F)
Ixy = cv2.GaussianBlur(Ixy,(5,5),1.0,cv2.CV_32F)

det_M = Ix2 * Iy2 - Ixy **2
trace_M = Ix2 + Iy2

R = det_M - 0.04 * (trace_M **2)
maxR = np.max(R)
thres = np.array(R>0.01*maxR)
R = R * thres
corners = np.argwhere(thres)
output = img.copy()
for corner in corners:
    row,col = corner
    cv2.circle(output,(col,row),4,(0,0,255),-1)

res_path = os.path.join(paths["results"],"harris.jpg")
verif = cv2.imwrite(res_path,output)
if verif:
    cv2.imshow("Harris: ",output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()