import numpy as np
import cv2 as cv
import os

def loadTrainData():
    train_data = []
    for dirname in os.listdir('Music_Symbols'):
        for filename in os.listdir('Music_Symbols/' + dirname):
            train_data.append(cv.imread('Music_Symbols/' + dirname + '/' + filename, 0))
    return train_data

#train_data = loadTrainData()
img = cv.imread('Music_Symbols/CLEF_Trebble/sol_agata_BN._1.bmp', 0)

def convolution(img):
    out = []
    # Filtro para bordas verticais 
    filter = np.array([[-1,  0,  1],
                       [-1,  0,  1],
                       [-1,  0,  1]])
    out.append(cv.filter2D(img, -1, filter))
    # Filtro para
    filter = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    out.append(cv.filter2D(img, -1, filter))
    # Filtro para bordas horizontais
    filter = np.array([[-1, -1, -1],
                       [ 0,  0,  0],
                       [ 1,  1,  1]])
    out.append(cv.filter2D(img, -1, filter))
    return out

o = convolution(img)
tot = o[0]
for conc in o:
    tot = np.concatenate((tot, conc), axis=1)
cv.imshow('image', tot)
k = cv.waitKey(0)
if k == 27:
    cv.destroyAllWindows()
print(img)
