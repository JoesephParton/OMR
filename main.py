import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from imutils.perspective import four_point_transform


def display_img(path_img):
    img = np.asarray(Image.open(path_img))
    imgplot = plt.imshow(img)
    plt.show()
    

def get_rect_cnts(contours): # a function is used to allow reuse of the code
    rect_cnts = []
    for cnt in contours:
        # approximate the contour
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        # if the approximated contour is a rectangle ...
        if len(approx) == 4:  # looking for 4 points/corners of the same intensity
            # append it to our list
            rect_cnts.append(approx)
    # sort the contours from biggest to smallest
    rect_cnts = sorted(rect_cnts, key=cv2.contourArea, reverse=True)

    return rect_cnts


#load img
display_img("images/2.jpg")

img = cv2.imread("images/2.jpg")
correct = ["A", "B", "C", "D", "E"]

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

ret, thresh = cv2.threshold(blur_img, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
edge_img = cv2.Canny(blur_img, 10, 70)
contours, _ = cv2.findContours(edge_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cntr = get_rect_cnts(contours)
document = four_point_transform(img, cntr[0].reshape(4, 2))
# contor_img = cv2.drawContours(blur_img, contours, 1, (0,255,0), 3)


x,y,w,h = cv2.boundingRect(contours[2])
cropped_img = blur_img[y:y+h, x:x+w]

cv2.imwrite('images/2_grey.jpg', gray_img)
cv2.imwrite('images/2_blur.jpg', blur_img)
cv2.imwrite('images/2_edge.jpg', edge_img)
cv2.imwrite('images/2_cont.jpg', contor_img)
cv2.imwrite('images/2_crop.jpg', cropped_img)
cv2.imwrite('images/2_crnt.jpg', document)

display_img('images/2_grey.jpg')
display_img('images/2_blur.jpg')
display_img('images/2_edge.jpg')
display_img('images/2_cont.jpg')
display_img('images/2_crop.jpg')
display_img('images/2_crnt.jpg')

