import numpy as np

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