import cv2
import matplotlib.pyplot as plt
import numpy as np

im = cv2.imread(r'img\barcodes.jpg', cv2.IMREAD_GRAYSCALE)

scale = 800.0 / im.shape[1]
im = cv2.resize(im, (int(im.shape[1] * scale), int(im.shape[0] * scale)))

kernel = np.ones((1, 3), np.uint8)
im = cv2.morphologyEx(im, cv2.MORPH_BLACKHAT, kernel, anchor=(1, 0))

thresh, im = cv2.threshold(im, 10, 255, cv2.THRESH_BINARY)

plt.imshow(im, cmap='Greys_r')
