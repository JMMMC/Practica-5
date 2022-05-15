import cv2
import numpy as np
import imutils

#se obtine la imagen
img = cv2.imread('Texto.jpeg',0)

# Escalando una imagenes usando imutils.resize
rimg = imutils.resize(img,width = 700)  #1520 el total

#####################  Thresholds  ##########################################################################################################################################################

#Threshold binary
ret,bin_ = cv2.threshold(rimg, 20, 255, cv2.THRESH_BINARY)

#Threshold binary_inv,
ret,bin_inv = cv2.threshold(rimg, 20, 255, cv2.THRESH_BINARY_INV)

#Threshold Trunc
ret,trunc = cv2.threshold(rimg, 50, 255, cv2.THRESH_TRUNC)

#Threshold To Zero
ret,to0 = cv2.threshold(rimg, 50, 255, cv2.THRESH_TOZERO)

#Threshold To Zero_inv
ret,to0_inv = cv2.threshold(rimg, 50, 255, cv2.THRESH_TOZERO_INV)

#Threshold Mean
mean = cv2.adaptiveThreshold(rimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5)

#Threshold Gauss
gauss = cv2.adaptiveThreshold(rimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)

#Threshold Otsu.
ret,otsu = cv2.threshold(rimg,120,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 

########################  Mostrar en Pantalla  ##########################################################################################################################################################


#Mostramos Izquierda en Pantalla
winname = "Original"
cv2.namedWindow(winname)
cv2.moveWindow(winname, 30,200)
cv2.imshow(winname, rimg)

winname = 'Binario'
cv2.namedWindow(winname)
cv2.moveWindow(winname, 790,200)
cv2.imshow(winname, bin_)
cv2.waitKey(0)
cv2.destroyWindow(winname)

winname = 'Binario Invertido'
cv2.namedWindow(winname)
cv2.moveWindow(winname, 790,200)
cv2.imshow(winname, bin_inv)
cv2.waitKey(0)
cv2.destroyWindow(winname)

winname = 'Truncado'
cv2.namedWindow(winname)
cv2.moveWindow(winname, 790,200)
cv2.imshow(winname, trunc)
cv2.waitKey(0)
cv2.destroyWindow(winname)

winname = 'A cero'
cv2.namedWindow(winname)
cv2.moveWindow(winname, 790,200)
cv2.imshow(winname, to0)
cv2.waitKey(0)
cv2.destroyWindow(winname)

winname = 'A cero invertido'
cv2.namedWindow(winname)
cv2.moveWindow(winname, 790,200)
cv2.imshow(winname, to0_inv)
cv2.waitKey(0)
cv2.destroyWindow(winname)

winname = 'Mean'
cv2.namedWindow(winname)
cv2.moveWindow(winname, 790,200)
cv2.imshow(winname, mean)
cv2.waitKey(0)
cv2.destroyWindow(winname)

winname = 'Gauss'
cv2.namedWindow(winname)
cv2.moveWindow(winname, 790,200)
cv2.imshow(winname, gauss)
cv2.waitKey(0)
cv2.destroyWindow(winname)

winname = 'Otsu'
cv2.namedWindow(winname)
cv2.moveWindow(winname, 790,200)
cv2.imshow(winname, otsu)

##Final## 
cv2.waitKey(0)
cv2.destroyAllWindows()
