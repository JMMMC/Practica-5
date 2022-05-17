import cv2
import numpy as np
import imutils

#se obtine la imagen
img = cv2.imread('Texto.jpeg',0)

# Escalando una imagenes usando imutils.resize
rimg = imutils.resize(img,width = 700)  #1520 el total

#####################  Impresion  ##########################################################################################################################################################

def Impresion(namme,imagen,x,y):
    cv2.namedWindow(namme)
    cv2.moveWindow(namme, x,y)
    cv2.imshow(namme, imagen)

def Impresion_destroy(namme,imagen,x,y):
    cv2.namedWindow(namme)
    cv2.moveWindow(namme, x,y)
    cv2.imshow(namme, imagen)
    cv2.waitKey(0)
    cv2.destroyWindow(namme)

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

Impresion('Original',rimg,30,200)
Impresion_destroy('Binario',bin_,790,200)
Impresion_destroy('Binario Invertido',bin_inv,790,200)
Impresion_destroy('Truncado',trunc,790,200)
Impresion_destroy('A cero',to0,790,200)
Impresion_destroy('A cero invertido',to0_inv,790,200)
Impresion_destroy('Mean',mean,790,200)
Impresion_destroy('Gauss',gauss,790,200)
Impresion('Otsu',otsu,790,200)

##Final## 
cv2.waitKey(0)
cv2.destroyAllWindows()
