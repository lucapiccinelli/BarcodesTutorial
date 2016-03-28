import cv2
import matplotlib.pyplot as plt

im = cv2.imread(r'img\barcodes.jpg', cv2.IMREAD_GRAYSCALE)

plt.imshow(im, cmap='Greys_r')
