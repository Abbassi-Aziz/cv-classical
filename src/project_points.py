import numpy as np

k = np.array([[800,0,320],
              [0,800,240],
              [0,0,1]])

point = (0.5 , 0.3 , 2.0)

u = k[0][0]*point[0]/point[2] + k[0][2]
v = k[1][1]*point[1]/point[2] + k[1][2]

print(u)
print(v)