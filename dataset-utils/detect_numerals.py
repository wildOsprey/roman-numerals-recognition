import cv2
import numpy as np

im = cv2.imread(r'..//dataset//original//Image (2).png')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#http://opencvexamples.blogspot.com/2013/10/applying-bilateral-filter.html
gray = cv2.bilateralFilter(gray, 20, 75, 75)
edged = cv2.Canny(gray, 10, 256)

contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

bxs = []
for cnt in contours:
    [x, y, w, h] = cv2.boundingRect(cnt)
    # print("contour:" w, h)
    if h > 30:
        bxs.append([x, y, w, h])
        cv2.rectangle(im, (x - 1, y - 1), (x + 1 + w, y + 1 + h), (0, 0, 255), 1)
        roi = edged[y:y + h, x:x + w]
        roismall = cv2.resize(roi, (10, 10))


def findCloses(box, boxes, threshold):
    for another_box in boxes:
        if another_box != box:
            if abs(box[0] + box[2] - another_box[0]) < threshold or abs(another_box[0] + another_box[2] - box[0]) < threshold:
                return another_box

    return None

    
boxes = bxs
it = 0
while True:
    it+=1
    updates_boxes = []
    print(it)
    for box in boxes:
        closest_box = findCloses(box, boxes, 40)
        if closest_box != None:
            x = min(closest_box[0], box[0])
            y = min(closest_box[1], box[1])
            if closest_box[0] < box[0]:
                w = abs(closest_box[0] + closest_box[2] - box[0])
            else:
                w = abs(box[0] + box[2] - closest_box[0])
            
            h = max(closest_box[3], box[3])
            updates_boxes.append([x, y, w, h])

    if len(updates_boxes) == len(boxes) or it > 100:
        break

    boxes = updates_boxes

for box in boxes:
    x, y, w, h = box  
    cv2.rectangle(im, (x - 1, y - 1), (x + 1 + w, y + 1 + h), (0, 255, 255), 1)
    

cv2.imshow('Training: Enter digits displayed in the red rectangle!', im)
key = cv2.waitKey(0)



