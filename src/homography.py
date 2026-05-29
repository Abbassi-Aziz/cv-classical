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

if len(good) > 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    res_path = os.path.join(paths["results"],"warped.jpg")
    warped_img = cv2.warpPerspective(img, H, (other.shape[1], other.shape[0]))
    verif = cv2.imwrite(res_path, warped_img)
    if verif:
        cv2.imshow("Warped Image: ", warped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()