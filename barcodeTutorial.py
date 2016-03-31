import cv2
import matplotlib.pyplot as plt
import numpy as np

im = cv2.imread(r'img\barcodes.jpg', cv2.IMREAD_GRAYSCALE)

#riscalatura dell'immagine
scale = 800.0 / im.shape[1]
im = cv2.resize(im, (int(im.shape[1] * scale), int(im.shape[0] * scale)))

#blackhat
kernel = np.ones((1, 3), np.uint8)
im = cv2.morphologyEx(im, cv2.MORPH_BLACKHAT, kernel, anchor=(1, 0))

#sogliatura
thresh, im = cv2.threshold(im, 10, 255, cv2.THRESH_BINARY)

#operazioni  morfologiche
kernel = np.ones((1, 5), np.uint8)
im = cv2.morphologyEx(im, cv2.MORPH_DILATE, kernel, anchor=(2, 0), iterations=2) #dilatazione
im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=2)  #chiusura

kernel = np.ones((21, 35), np.uint8)
im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel, iterations=1)

plt.imshow(im, cmap='Greys_r')

