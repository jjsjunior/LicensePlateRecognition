import numpy as np
import cv2
import imutils
from PIL import Image


def firstCrop(img, predictions):
    predictions.sort(key=lambda x: x.get('confidence'))
    xtop = predictions[-1].get('topleft').get('x')
    ytop = predictions[-1].get('topleft').get('y')
    xbottom = predictions[-1].get('bottomright').get('x')
    ybottom = predictions[-1].get('bottomright').get('y')
    firstCrop = img[ytop:ybottom, xtop:xbottom]
    cv2.rectangle(img ,(xtop ,ytop) ,(xbottom ,ybottom) ,(0 ,255 ,0) ,3)
    return firstCrop

def secondCrop(img):
    gray =cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
    ret ,thresh = cv2.threshold(gray ,127 ,255 ,0)
    im2, contours, hierarchy = cv2.findContours(thresh ,cv2.RETR_LIST ,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    if(len(areas )!=0):
        max_index = np.argmax(areas)
        cnt =contours[max_index]
        x ,y ,w ,h = cv2.boundingRect(cnt)
        cv2.rectangle(img ,(x ,y) ,( x +w , y +h) ,(0 ,255 ,0) ,2)
        secondCrop = img[y: y +h ,x: x +w]
    else:
        secondCrop = img
    return secondCrop

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged






