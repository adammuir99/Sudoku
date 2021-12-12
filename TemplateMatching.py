from time import perf_counter_ns
import cv2
import numpy as np
import imutils
import pytesseract
import Sudoku

input_board = [[0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0],
               [0,0,0,0,0,0,0,0,0]]

# First match the image with the sudoku board template
# to locate the sudoku board within the template ###########################################################################################
# Read the template
template = cv2.imread('template.jpg')
# Convert the template to grayscale
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
# Get the Canny edges for increased accuracy
template = cv2.Canny(template, 50, 200)

# Store width and height of template
(tH, tW) = template.shape[:2]

# Display the template's canny edges
cv2.imshow("Template", template)
cv2.waitKey(0)

# Now begin multiscaling the image
# Read the main image
img_rgb = cv2.imread('pic.jpg')
 
# Convert it to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Keep track of the region and scale of the image with the best match
found = None

# Loop over the scales of the image
for scale in np.linspace(0.2, 1.0, 40)[::-1] :
    # resize the image according to the scale, and keep track
	# of the ratio of the resizing
	resized = imutils.resize(img_gray, width = int(img_gray.shape[1] * scale))
	r = img_gray.shape[1] / float(resized.shape[1])

	# if the resized image is smaller than the template, then break
	# from the loop
	if resized.shape[0] < tH or resized.shape[1] < tW:
		break

    # detect edges in the resized, grayscale image and apply template
	# matching to find the template in the image
	edged = cv2.Canny(resized, 10, 50)
	result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)
	(_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

	# Visualize the iteration
	# draw a bounding box around the detected region
	clone = np.dstack([edged, edged, edged])
	cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
		(maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
	cv2.imshow("Visualize", clone)
	cv2.waitKey(0)

	# if we have found a new maximum correlation value, then update
	# the bookkeeping variable
	if found is None or maxVal > found[0]:
		found = (maxVal, maxLoc, r)

# unpack the bookkeeping variable and compute the (x, y) coordinates
# of the bounding box based on the resized ratio
(_, maxLoc, r) = found
(startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
(endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

# draw a bounding box around the detected result and display the image
cv2.rectangle(img_rgb, (startX, startY), (endX, endY), (0, 0, 255), 2)
cv2.imshow("Image", img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Next we will detect the numbers that are found within the bounding box #################################################################
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Crop the image to only read the sudoku box
img_crop = img_rgb[startY:endY, startX:endX].copy()

# Resizing to a larger image will help tesseract detect the numbers
img_crop = cv2.resize(img_crop, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)

# Tesseract only works with RGB images, OpenCV uses BGR
img_text = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)

hImg, wImg,_ = img_text.shape

# Define the size of each box in the sudoku grid
boxW = int(wImg/9)
boxH = int(hImg/9)

boxes = pytesseract.image_to_boxes(img_text, config="--psm 6 -c tessedit_char_whitelist=0123456789")

for b in boxes.splitlines() :
    b = b.split(' ')
    print(b)
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    cv2.rectangle(img_text, (x,hImg-y), (w,hImg-h), (0,0,255), 1)
    cv2.putText(img_text, b[0], (x, hImg-y+25), cv2.FONT_HERSHEY_PLAIN, 1, (50, 50, 255), 2)

    # Get the coordinates of the found digit
    centerx = int((x+w)/2)
    centery = int(hImg - (y+h)/2)

    # Check which row the digit is in
    row = 0
    for row_ in range(9) :
        if centerx < (row_ * boxW + boxW) :
            row = row_
            break
    
    # Check which col the digit is in
    col = 0
    for col_ in range(9) :
        if centery < (col_ * boxH + boxH) :
            col = col_
            break
    
    # Enter the digit into our board
    input_board[col][row] = int(b[0])

# Display the detected board in the terminal
print("\nThe input board is:")
print(np.matrix(input_board))

# Show the detected digits
cv2.imshow("Text", img_text)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Solve the sudoku puzzle
print("\nThe solved board is:")
Sudoku.solve(input_board)

# Remove unnecessary characters from text document
with open(r'Solution.txt', 'r') as infile, \
     open(r'Formatted_Solution.txt', 'w') as outfile:
    data = infile.read()
    data = data.replace("[", " ")
    data = data.replace("]", " ")
    outfile.write(data)

# Generate an array from the text document
solved_board = np.genfromtxt("Formatted_Solution.txt", dtype=int)

# Iterate through the rows and columns, entering the solutions into the empty spaces
for row in range (9) : 
    for col in range (9) :
        Xcoord = int(row * boxW + 0.25 * boxW)
        Ycoord = int(col * boxH + 0.75 * boxH)

        if input_board[col][row] == 0:
            cv2.putText(img_crop, str(solved_board[col][row]), (Xcoord, Ycoord), cv2.FONT_HERSHEY_DUPLEX, 3, (50, 50, 255), 2)

cv2.imshow("Solution", img_crop)

cv2.waitKey(0)
cv2.destroyAllWindows()