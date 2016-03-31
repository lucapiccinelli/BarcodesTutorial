import cv2
import matplotlib.pyplot as plt
import numpy as np

im = cv2.imread(r'img\barcodes.jpg', cv2.IMREAD_GRAYSCALE)
im_out = cv2.imread(r'img\barcodes.jpg')

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

#estrazione dei componenti connessi
contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

unscale = 1.0 / scale
if contours != None:
    for contour in contours:
        
        # se l'area non è grande a sufficienza la salto 
        if cv2.contourArea(contour) <= 2000:
            continue
        
        #estraggo il rettangolo di area minima (in formato (centro_x, centro_y), (width, height), angolo)
        rect = cv2.minAreaRect(contour)
        #l'effetto della riscalatura iniziale deve essere eliminato dalle coordinate rilevate
        rect = \
            ((int(rect[0][0] * unscale), int(rect[0][1] * unscale)), \
             (int(rect[1][0] * unscale), int(rect[1][1] * unscale)), \
             rect[2])
        
        
        #disegno il tutto sull'immagine originale
        box = np.int0(cv2.cv.BoxPoints(rect))
        cv2.drawContours(im_out, [box], 0, (0, 255, 0), thickness = 2)
        
plt.imshow(im_out)
cv2.imwrite(r'img\out.png', im_out)