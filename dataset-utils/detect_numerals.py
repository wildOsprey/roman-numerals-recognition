import cv2
import numpy as np
from nms import non_max_suppression_fast
import random
import string

PATH = r'/home/alexandra/repos/roman_numerals_recognition/dataset/original/Image (3).png'
CLOSEST_THRESHOLD = 10
NMS_THRESHOLD = 0.4 

def findClosest(box, boxes, threshold):
    for another_box in boxes:
        if another_box != box:
            if abs(box[2] - another_box[0]) < threshold or abs(another_box[2] - box[0]) < threshold:
                return another_box

            if abs(box[3]- another_box[1]) < threshold or abs(another_box[3] - box[1]) < threshold:
                return another_box
    return None

def get_boxes(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    #http://opencvexamples.blogspot.com/2013/10/applying-bilateral-filter.html
    gray = cv2.bilateralFilter(gray, 20, 75, 75)
    edged = cv2.Canny(gray, 10, 256)

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    bxs = []

    for cnt in contours:
        [x, y, w, h] = cv2.boundingRect(cnt)
        if h >= 28:
            bxs.append([x - 15, y - 15, x+ max(w, h) + 15, y+max(w, h) + 15])

    boxes = bxs

    # while True:
    #     updates_boxes = []

    #     for box in boxes:
    #         closest_box = findClosest(box, boxes, CLOSEST_THRESHOLD)
    #         if closest_box != None:
    #             x = min(closest_box[0], box[0])
    #             y = min(closest_box[1], box[1])
    #             w = max(closest_box[2], box[2]) - x
    #             h = max(closest_box[3], box[3]) - y
                
    #             updates_boxes.append([x, y, x+w, y+h])
    #         else:
    #             updates_boxes.append(box)

    #     if len(updates_boxes) == len(boxes):
    #         break

    #     boxes = updates_boxes


    # merged_boxes = non_max_suppression_fast(updates_boxes, NMS_THRESHOLD)

    return boxes

def save_imgs(boxes, im):
    img_copy = im.copy()
    for box in boxes:
        x_top, y_top, x_bottom, y_bottom = box
        roi = im[ y_top:y_bottom,x_top:x_bottom, :]
        name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        cv2.imwrite('results/' + name + '.jpg', roi)
        cv2.rectangle(img_copy, (x_top - 1, y_top - 1), (x_bottom, y_bottom), (0, 0, 255), 1)

    cv2.imwrite('Original_' +''.join(random.choices(string.ascii_uppercase, k=3)) + '.jpg', img_copy)
    

img = cv2.imread(PATH)
boxes = get_boxes(img)
save_imgs(boxes, img) 

