import cv2
import os
import numpy as np 

def load_gray_img(path):
    assert os.path.exists(path), 'Img does not exist'
    img = cv2.imread(path, 0)
    return img

def inverse(img):
    return cv2.bitwise_not(img)

def resize(img):
    return cv2.resize(img, (48, 48))

def show_img(img):
    cv2.imshow('img', img)
    cv2.waitKey()

def contrast(img):
    (thresh, gray) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return gray

def save_img(path, img):
    cv2.imwrite(path, img)


def w2b(img):
    b = 0
    w = 0
    for row in img:
        for pixel in row:
            if pixel == 0:
                b+=1
            else: 
                w+=1
    return w / b * 100.

def get_img(path):
    img = load_gray_img(path)
    show_img(img)
    img = inverse(img)
    img = contrast(img)
    img = resize(img)
    show_img(img)
    per= w2b(img)
    print(per)
    save_img('../test_processed.jpg', img)

get_img('../test2.png')