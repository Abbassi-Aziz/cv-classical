import numpy as np
import cv2
from cvlib.filters import conv_2d

def harris(image, k=0.04, threshold = 0.01):
    Gx = np.array([[-1,0,1],
               [-2,0,2],
               [-1,0,1]], dtype=np.float32)
    Gy = np.array([[-1,-2,-1],
               [0,0,0],
               [1,2,1]], dtype=np.float32)

    Ix = conv_2d(image, Gx)
    Iy = conv_2d(image,Gy)

    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    Ix2 = cv2.GaussianBlur(Ix2,(5,5),1.0)
    Iy2 = cv2.GaussianBlur(Iy2,(5,5),1.0)
    Ixy = cv2.GaussianBlur(Ixy,(5,5),1.0)

    det_M = Ix2 * Iy2 - Ixy **2
    trace_M = Ix2 + Iy2

    R = det_M - k * (trace_M **2)
    maxR = np.max(R)
    thres = np.array(R> threshold*maxR)
    R = R * thres
    corners = np.argwhere(thres)
    return corners, R

def sift_detect_match(img1, img2,ratio=0.75):
    sift = cv2.SIFT_create()
    kp1 , des1 = sift.detectAndCompute(img1, None)
    kp2 , des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m,n in matches:
        if m.distance / n.distance < ratio :
            good.append(m)
    return kp1, kp2, good