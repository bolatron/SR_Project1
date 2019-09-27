import numpy as np
import cv2 as cv

def convolution(img):
    out = []
    # Filtro para bordas verticais (SobelV)
    filter = np.array([[-1,  0,  1],
                       [-2,  0,  2],
                       [-1,  0,  1]])
    out.append(cv.filter2D(img, -1, filter))
    # Filtro para
    filter = np.array([[-1, -1, -1],
                       [-1,  8, -1],
                       [-1, -1, -1]])
    out.append(cv.filter2D(img, -1, filter))
    filter = np.array([[ 1,  1,  1],
                       [ 1, -8,  1],
                       [ 1,  1,  1]])
    out.append(cv.filter2D(img, -1, filter))
    # Filtro para bordas horizontais (SobelH)
    filter = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]])
    out.append(cv.filter2D(img, -1, filter))
    return out

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.blur(gray, (3,3))
    #out = convolution(gray)
    #o = out[1]
    #o = np.concatenate((o, out[2]), axis=1)
    threshold = 50
    canny_output = cv.Canny(gray, threshold, threshold * 2)
    # Display the resulting frame
    cv.imshow('frame', canny_output)
    if cv.waitKey(1) == 27:
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
