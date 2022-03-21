from __future__ import print_function, division 

import cv2
import os,sys, os.path
from cv2 import HOUGH_GRADIENT
import numpy as np
import argparse
from hough_helper import mostra_imagem



def magnitude_do_gradiente(img, showfig=False):
    
    small = cv2.resize(img, (800, 1960), 0,0, interpolation = cv2.INTER_AREA)
      # Filtro de Sobel para a derivada ao longo de X
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0)

    # Filtro de Sobel para a derivada ao longo de Y
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1)
    
    # Magnitude do gradiente
    mag_gradiente_1 = (sobelx**2 + sobely**2)**0.5
    
    # Converte a imagem de ponto flutuante de 64 bits para imagem de 8 bits
    mag_gradiente_2 = cv2.convertScaleAbs(mag_gradiente_1)
    min_contrast = 100
    max_contrast = 200
    linhas = cv2.Canny(small, min_contrast, max_contrast ) 
    return mag_gradiente_2, linhas