import cv2
import numpy as np
import os
import time 

BASE = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(BASE, "../data/butterfly.jpg")

img = cv2.imread(img_path)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

def conv_2d (image , kernel):
    h,w = image.shape
    kh, kw = kernel.shape
    kh_half , kw_half = kh//2 , kw//2
    output = np.zeros_like(image,dtype=np.float32)
    for i in range(kh_half,h-kh_half):
        for j in range(kw_half,w-kw_half):
            total = 0
            for m in range(kh):
                for n in range(kw):
                    total += image[i - kh_half +m , j - kw_half +n] * kernel[m,n]
            output[i][j] = total
    return output

blur = np.ones((3,3),dtype=np.float32) / 9.0
sharpen = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=np.float32)

blurred = conv_2d(gray, blur)
sharpened = conv_2d(gray, sharpen)

blurred_clipped = np.clip(blurred ,0,255).astype(np.uint8)
sharpened_clipped = np.clip(sharpened,0,255).astype(np.uint8)

cv2.imshow("blurred",blurred_clipped)
cv2.imshow("sharpened",sharpened_clipped)

os.makedirs("output", exist_ok=True)
cv2.imwrite("output/blurred.jpg", blurred_clipped)
cv2.imwrite("output/sharpened.jpg", sharpened_clipped)
cv2.imwrite("output/original.jpg", gray)

cv2.waitKey(0)
cv2.destroyAllWindows()

start = time.time()
blurred_manual = conv_2d(gray, blur)
manual_time = time.time() - start

start = time.time()
blurred_cv = cv2.filter2D(gray, -1, blur)
cv_time = time.time() - start

print(f"Manual: {manual_time:.3f}s")
print(f"OpenCV: {cv_time:.3f}s")
print(f"Speedup: {manual_time/cv_time:.0f}x")