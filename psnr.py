import cv2
import numpy as np

# Read images
original = cv2.imread("hazy.png")
enhanced = cv2.imread("SSR_result.png")

# Check same size
print(original.shape)
print(enhanced.shape)

# Calculate MSE
mse = np.mean((original.astype(np.float64) -
               enhanced.astype(np.float64)) ** 2)

# Calculate PSNR
if mse == 0:
    psnr = float('inf')
else:
    psnr = 10 * np.log10((255 ** 2) / mse)

print("MSE  =", mse)
print("PSNR =", psnr, "dB")