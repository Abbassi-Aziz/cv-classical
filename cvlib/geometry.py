import numpy as np
from cvlib.features import sift_detect_match
import cv2

def project_point(K, point_3d):
    projected = K @ point_3d
    u = projected[0] / projected[2]
    v = projected[1] / projected[2]
    return u, v

def compute_homography(img1,img2):
    kp1,kp2,good = sift_detect_match(img1,img2)
    if len(good) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
        warped_img = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]))

        return H, mask, warped_img
    
    else:
        raise ValueError(f"Not enough good matches: {len(good)}, need at least 4")
