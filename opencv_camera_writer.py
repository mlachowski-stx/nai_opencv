from collections import deque

import cv2
import numpy as np
import pytesseract
import re
import time
from PIL import Image

# Define the upper and lower boundaries for a color to be considered "Blue"
blueLower = np.array([52, 142, -7])
blueUpper = np.array([152, 242, 193])

# Define a 5x5 kernel for erosion and dilation
kernel = np.ones((5, 5), np.uint8)

bpoints = [deque(maxlen=512)]
bindex = 0

paintColor = (0, 0, 0)
textColor = (255, 255, 255)

paintWindow = np.zeros((471, 636, 3)) + 255

clearButtonRectangleArgs = ((40, 1), (140, 65), (122, 122, 122), -1)
clearButtonArgs = ("CLEAR", (63, 38), cv2.FONT_HERSHEY_DUPLEX, 0.5, textColor, 2, cv2.LINE_AA)

cv2.namedWindow('Output', cv2.WINDOW_AUTOSIZE)

camera = cv2.VideoCapture(0)

start_time = time.time()
letters = ''
regex = re.compile('[A-Z0-9]')
config = ("-l eng --oem 1 --psm 7")


def recognize_text(painting, bpoints, letters):
    imgArray = paintWindow.astype('uint8')
    img = Image.fromarray(imgArray)
    text = pytesseract.image_to_string(img, config=config)
    if len(text) == 1 and regex.match(text) is not None:
        letters += text
        print(letters)
    else:
        print('Letter not recognized')

    return clear_points(), letters


def clear_points():
    bpoints = [deque(maxlen=512)]
    paintWindow[:, :, :] = 255
    return bpoints


while True:
    # Grab the current paintWindow
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if not grabbed:
        break

    # Add the same paint interface to the camera feed captured through the webcam (for ease of usage)
    frame = cv2.rectangle(frame, *clearButtonRectangleArgs)
    cv2.putText(frame, *clearButtonArgs)

    cv2.putText(frame, letters, (183, 38), cv2.FONT_HERSHEY_DUPLEX, 0.5, textColor, 2, cv2.LINE_AA)

    # Determine which pixels fall within the blue boundaries and then blur the binary image
    blueMask = cv2.inRange(hsv, blueLower, blueUpper)
    blueMask = cv2.erode(blueMask, kernel, iterations=2)
    blueMask = cv2.morphologyEx(blueMask, cv2.MORPH_OPEN, kernel)
    blueMask = cv2.dilate(blueMask, kernel, iterations=1)

    # Find contours in the image
    _, cnts, _ = cv2.findContours(blueMask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check to see if any blue contours were found
    if len(cnts) > 0:
        # Sort the contours and find the largest one
        cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

        if cv2.contourArea(cnt) >= 2000:
            # Get the radius of the enclosing circle and draw it
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

            # Get the moments to calculate the center of the circle
            M = cv2.moments(cnt)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

            if center[1] <= 65:
                if 40 <= center[0] <= 140:  # Clear All
                    bpoints = clear_points()
            else:
                bpoints[bindex].appendleft(center)

            start_time = time.time()

    # Draw lines
    points = [bpoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], paintColor, 5)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], paintColor, 5)

    # Show the frame and the paintWindow image
    cv2.imshow("Tracking", frame)
    cv2.imshow("Output", paintWindow)

    if time.time() - start_time > 2:
        bpoints, letters = recognize_text(paintWindow, bpoints, letters)
        start_time = time.time()

    # If the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup code
camera.release()
cv2.destroyAllWindows()
