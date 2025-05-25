import cv2 as cv
import numpy as np

img = cv.imread('Photos/cat.jpg')

cv.imshow('Boston', img)

cv.waitKey(0)
cv.destroyAllWindows()