import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def display_img(path_img):
    # quickly takes an img path and outputs it using matplotlib
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


def get_sans(img, correct):
    student_ans = []

    height, width = img.shape
    leng = len(correct)
    v_split = int(round(height / leng))
    h_split = int(round(width / 5))

    top = 0
    bottum = v_split

    for _ in range(0, leng):
        top_w = 0
        bottum_w = h_split
        most_like = []
        last = 0

        for i in range(0, 5):
            cropped_s = img[top:bottum, top_w:bottum_w]
            for_col = np.sum(cropped_s <= 200)
            if for_col > last:
                last = for_col
                most_like = correct[i]

            top_w = bottum_w
            bottum_w += h_split

        top = bottum
        bottum += v_split
        student_ans.append(most_like)

    return student_ans

# load img
file_path = "images/3.jpg"
img = cv2.imread(file_path)

#init
options = ["A", "B", "C", "D", "E"]
correct = ["A", "C", "B", "D", "E"]

# Grey Scale img
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur img
blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

ret, thresh = cv2.threshold(blur_img, 127, 255, 0)

# makes edges more prominint
edge_img = cv2.Canny(blur_img, 10, 70)

# gets the contors of the pages
contours, _ = cv2.findContours(edge_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cntr = get_rect_cnts(contours)
contor_img = cv2.drawContours(blur_img, cntr, 1, (255,255,255), 25)

# crops to the grade countor box
x,y,w,h = cv2.boundingRect(cntr[1])
cropped_img = contor_img[y:y+h, x:x+w]

stu_ans = get_sans(cropped_img, options)
print(stu_ans)

correct_count = 0

for index, ans in enumerate(correct):
    if stu_ans[index] == ans:
        correct_count += 1

print(correct_count, "/", len(correct))

