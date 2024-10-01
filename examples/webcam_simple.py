###
### PRE-ORDER BOOK on DIY AI! https://a.co/d/eDxJXJ0
###

import numpy as np
import cv2 as cv

CAMERA_ID = 0 # if you have a usb camera this may be 1

def process_image_as_gray(image):
    return  cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def process_image_as_edges(image, lower_threshold = 300, upper_threshold = 300, aperture=3):
    image = np.array(image)
    image = cv.Canny(image, lower_threshold, upper_threshold,apertureSize=aperture)
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    return image

# initiate video capture class
cap = cv.VideoCapture(CAMERA_ID)

WIDTH  = cap.get(cv.CAP_PROP_FRAME_WIDTH) #320
HEIGHT = cap.get(cv.CAP_PROP_FRAME_HEIGHT) #240

cap.set(cv.CAP_PROP_FRAME_WIDTH, WIDTH/2) 
cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT/2)

print(cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT))

while True:

    # Capture frame-by-frame
    ret, image = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # result_frame = process_image_as_gray(image)
    result_frame = process_image_as_edges(image, lower_threshold = 100, upper_threshold = 100, aperture=3)

    cv.imshow('frame', result_frame)

    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()