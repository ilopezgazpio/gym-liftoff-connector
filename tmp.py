#read an image
import cv2

img = cv2.imread('gym-liftoff/gym_liftoff/main/screenshot.png')

# save bottom right corner of the image
bottom_right = img[180:220, 1780:1860]
#binarize image
gray = cv2.cvtColor(bottom_right, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
#display the image

cv2.imshow('image', binary)

#wait for a key press
cv2.waitKey(0)

#close the window

cv2.destroyAllWindows()

