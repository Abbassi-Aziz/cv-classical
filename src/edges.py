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

Gx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]], dtype=np.float32)
Gy = np.array([[-1,-2,-1],
               [ 0, 0, 0],
               [ 1, 2, 1]], dtype=np.float32)

out_x = conv_2d(gray,Gx)
out_y = conv_2d(gray,Gy)
M = np.sqrt(out_x**2+out_y**2)
M = cv2.normalize(M, None, 0 ,255, cv2.NORM_MINMAX).astype(np.uint8)

manual_path = os.path.join(paths["results"],"manual_edge.jpg")
opencv_path = os.path.join(paths["results"],"canny_edge.jpg")
verify = cv2.imwrite(manual_path,M)
if verify:
    cv2.imshow("manual_edge",M)

N = cv2.Canny(gray,80,190)
verify = cv2.imwrite(opencv_path,N)
if verify:
    cv2.imshow("opencv_edge",N)

cv2.waitKey(0)
cv2.destroyAllWindows()