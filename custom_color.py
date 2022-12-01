import cv2
import numpy as np
import colorsys

def custom_color(r, g, b, w1=0.225, w2=0.775, w3=1.0, w4=0.0):

    img = cv2.imread('img/custom/custom.png') # 1024x1024
    mask = cv2.imread('img/custom/mask.png') # 512x512
    # Resize mask to 1024x1024
    mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

    # Create a new image
    new_img = np.zeros_like(img)
    hsv_goal = colorsys.rgb_to_hsv(r, g, b)
    # Loop over the image
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            # Check if the pixel is purple
            # if np.array_equal(mask[i, j], [255, 102, 204]):
            if np.array_equal(mask[i, j], [255, 255, 255]):
                # Change the color to the new one
                temp = [x for x in img[i][j]]
                hsv_temp = list(colorsys.rgb_to_hsv(*temp))
                hsv_temp[0] = hsv_goal[0]
                # lets add a little bit of lightness
                hsv_temp[1] = w1*hsv_temp[1] + w2*hsv_goal[1]
                # lets add a little bit of darkness
                hsv_temp[2] = w3*hsv_temp[2] + w4*hsv_goal[2]

                new_img[i, j] = colorsys.hsv_to_rgb(*hsv_temp)
                
            else:
                # Keep the original color
                new_img[i, j] = img[i, j]
    # Save the new image
    cv2.imwrite('img/custom/new_hair_{}_{}_{}.png'.format(r, g, b), new_img)

# custom_color(0, 0, 255)


# Lets try some colors ---- GRID SEARCH can be done below
# r, g, b = 255, 255, 0
# weights1 = [0.225]
# weights2 = [1.0]

# for w1 in weights1:
#     for w3 in weights2:
#         custom_color(r, g, b, w1, 1 - w1, w3, 1 - w3)