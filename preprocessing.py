import time
import numpy as np
import cv2
import pytesseract
from PIL import Image
import imutils
import imghdr
import re



def preprocessing_img(img,x_start,x_end,y_start,y_end,angle_rot):
    """
    Возращает три изображения с разными параметрами фильтраций .Возможно повернуть исходное изображение
    :param img:
    :param x_start:
    :param x_end:
    :param y_start:
    :param y_end:
    :param angle_rot:
    :return:
    """
    img_gray2 = cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 199, 25)
    if angle_rot!=0:
        img_gray2_rot=imutils.rotate_bound(img_gray2,angle=270)
        img_gray2_obr = img_gray2_rot[x_start:x_end, y_start:y_end]
        # предобработка
        # медианная фильтрация
        img_gray2_obr1 = cv2.medianBlur(img_gray2_obr, 3)
        # замыливание
        blurred_img = cv2.GaussianBlur(img_gray2_obr1, ksize=(5, 5), sigmaX=0, sigmaY=0)
        return img_gray2_obr, img_gray2_obr1, blurred_img
    img_gray2_obr = img_gray2[x_start:x_end, y_start:y_end]
    # предобработка
    # медианная фильтрация
    img_gray2_obr1 = cv2.medianBlur(img_gray2_obr, 3)
    # замыливание
    blurred_img = cv2.GaussianBlur(img_gray2_obr1, ksize=(5, 5), sigmaX=0, sigmaY=0)
    return img_gray2_obr, img_gray2_obr1, blurred_img



