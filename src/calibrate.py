import cv2
import numpy as np
import os
from cvlib.utils import get_project_paths

paths = get_project_paths(__file__)
data_dir = paths['data']

images = []
for i in range(1,15):
    number = f'0{i}' if i<10 else f'{i}'
    if i != 10: 
        img_path = os.path.join(data_dir, f'calibration/left{number}.jpg')
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not load {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images.append(img)

objp = np.zeros((6*9, 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

objpoints = []
imgpoints = []

for img in images:
    ret, corners = cv2.findChessboardCorners(img, (9,6), None)
    if ret:
        objpoints.append(objp)
        refined = cv2.cornerSubPix(img, corners, (11,11), (-1,-1), criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(refined)

ret,k,d,rvecs,tvecs = cv2.calibrateCamera(objpoints, imgpoints, images[0].shape[::-1], None, None)
print("Camera Matrix:\n", k)
print("Distortion Coefficients:\n", d)
print("Mean Reprojection Error:", ret)

save_path = os.path.join(data_dir, 'calibration.npz')
np.savez(save_path, camera_matrix=k, dist_coeffs=d)

undistorted = cv2.undistort(images[0], k, d)
comparison = np.hstack([images[0], undistorted])
out_path = os.path.join(paths["results"], "undistorted_comparison.jpg")
cv2.imwrite(out_path, comparison)