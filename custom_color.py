import cv2



img_red = cv2.imread('custom/red.png')
img_green = cv2.imread('custom/green.png')
img_blue = cv2.imread('custom/blue.png')

def changer(new):

    (r, g, b) = new
    # let's change all red pixels to the r value of the new color
    red_pixels = (img_red[:, :, 0] == 255) & (img_red[:, :, 1] == 0) & (img_red[:, :, 2] == 0)
    img_red[:, :, :3][red_pixels] = [r, g, b]
    # let's change all green pixels to the g value of the new color
    green_pixels = (img_green[:, :, 0] == 0) & (img_green[:, :, 1] == 255) & (img_green[:, :, 2] == 0)
    img_green[:, :, :3][green_pixels] = [r, g, b]
    # let's change all blue pixels to the b value of the new color
    blue_pixels = (img_blue[:, :, 0] == 0) & (img_blue[:, :, 1] == 0) & (img_blue[:, :, 2] == 255)
    img_blue[:, :, :3][blue_pixels] = [r, g, b]
    # let's combine the three images
    img = cv2.add(img_red, img_green)
    img = cv2.add(img, img_blue)
    cv2.imwrite('custom/combined.png', img)


changer((35, 127, 48))
