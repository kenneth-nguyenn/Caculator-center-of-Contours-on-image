import imutils
import cv2

vid = cv2.VideoCapture(0)
while (True):
    ret, frame = vid.read()  
    if ret == True:
        # load the image, convert it to grayscale, blur it slightly, and threshold it
        image = frame
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

        cv2.imshow('gray', gray)
        cv2.imshow('blur', blurred)
        cv2.imshow('thresh', thresh)

        # find contours in the thresholded image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        # loop over the contours
        for c in cnts:
            # compute the center of the contour
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # draw the contour and center of the shape on the image
                cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
                cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
                cv2.putText(image, "center", (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imshow('Frame', frame)
    key = cv2.waitKey(20)
    # exit on ESC
    if key == 27:
        break

vid.release()
cv2.destroyWindow("frame")
