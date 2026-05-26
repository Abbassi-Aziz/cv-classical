import cv2
import numpy as np 
import os
from cvlib.utils import get_project_paths

paths = get_project_paths(__file__)
img_path = os.path.join(paths["data"],"butterfly.jpg")
other_img_path = os.path.join(paths["data"],"butterfly_modified.jpg")

img = cv2.imread(img_path)
other = cv2.imread(other_img_path)

sift = cv2.SIFT_create()
kp1 , des1 = sift.detectAndCompute(img, None)
kp2 , des2 = sift.detectAndCompute(other, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m,n in matches:
    if m.distance / n.distance < 0.75 :
        good.append(m)

result_img = cv2.drawMatches(img, kp1, other, kp2, good[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
res_path = os.path.join(paths["results"],"matches.jpg")
verif = cv2.imwrite(res_path,result_img)
if verif:
    cv2.imshow("Matches: ",result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()