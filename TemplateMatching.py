import cv2
import numpy as np
 
# Read the main image
img_rgb = cv2.imread('pic.jpg')
 
# Convert it to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
 
# Read the template
template = cv2.imread('template.jpg',0)
 
# Store width and height of template in w and h
w, h = template.shape[::-1]
 
# Perform match operations.
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
 
# Specify a threshold
threshold = 0.4

if np.max(res) < threshold :
    print("No Sudoku grid found in image.")
    exit()
 
# Tuple loc holds the minimum and maximum matches and their respective coordinates
loc = cv2.minMaxLoc(res)    #loc[0] -> minimum value
                            #loc[1] -> maximum value
                            #loc[2] -> coordinates of minimum
                            #loc[3] -> coordinates of maximum
 
# Draw a rectangle around the matched region.
cv2.rectangle(img_rgb, loc[3], (loc[3][0] + w, loc[3][1] + h), (0,0,255), 2)
 
# Show the final image with the matched area.
cv2.imshow('Detected',img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()