import cv2
import numpy as np
from cvlib.filters import conv_2d

def sobel(image):
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Gy = np.array([[-1,-2,-1],
                   [ 0, 0, 0],
                   [ 1, 2, 1]], dtype=np.float32)

    out_x = conv_2d(image,Gx)
    out_y = conv_2d(image,Gy)
    magnitude = np.sqrt(out_x**2+out_y**2)
    #M = cv2.normalize(M, None, 0 ,255, cv2.NORM_MINMAX).astype(np.uint8)
    direction = np.arctan2(out_y, out_x) 
    
    return out_x, out_y, magnitude, direction

def canny(image, t_low, t_high):
    # TODO: Implement the Canny edge detection algorithm
    pass
    