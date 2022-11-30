import cv2
import numpy as np



img_red = cv2.imread('custom/red.png')
# img_green = cv2.imread('custom/green.png')
# img_blue = cv2.imread('custom/blue.png')

def changer(new):

    (r, g, b) = new
    # let's change rgb to hsv to make it easier to change the color
    hsv = cv2.cvtColor(img_red, cv2.COLOR_BGR2HSV)
    # find the red parts in the image
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    # change the color of the red parts
    hsv[mask > 0] = ([r, g, b])
    # change back to rgb
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('custom/red_.png', img)


changer((35, 127, 48))
