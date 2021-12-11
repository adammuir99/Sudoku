import numpy as np
import cv2

image1 = cv2.imread('pic.jpg')
gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
canimg = cv2.Canny(gray, 50, 200)

lines = cv2.HoughLines(canimg, 1, np.pi/180.0, 200, np.array([]))

for line in lines : 
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(image1, (x1, y1), (x2, y2), (0,0,255), 2)

cv2.imshow('Lines Detected', image1)
cv2.imshow("Canny Detection", canimg)
cv2.waitKey(0)
cv2.destroyAllWindows()