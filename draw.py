import cv2 as cv
import numpy as np

blank = np.zeros((500, 500, 3), dtype='uint8')
cv.imshow('Blank', blank)


blank[200:300] = 0,255,0
cv.imshow('Green', blank)

cv.rectangle(blank, (0, 0), (250, 250), (0, 255, 0), thickness=cv.FILLED)
cv.imshow('Rectangle', blank)

cv.line (blank,(0, 0), (blank.shape[1]//2, blank.shape[0]//2), (255, 0, 0), thickness=3)
cv.imshow('Line', blank)


cv.putText(blank, 'Hello', (225,225),cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 0, 225), 2)
cv.imshow('Text', blank)


cv.waitKey(0)