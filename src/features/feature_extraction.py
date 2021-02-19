import numpy as np
import cv2
from math import ceil, floor, sqrt

def resize_img(img, new_size):
    return cv2.resize(img, new_size, cv2.INTER_AREA)

def pad_image(img):
    new_img = img
    h, w, c = img.shape
    dif = abs(h - w)
    if h > w:
        left = np.zeros((h, ceil(dif/2), c))
        right = np.zeros((h, floor(dif/2), c))
        new_img = np.concatenate((left, img, right), 1)
    elif w > h:
        up = np.zeros((ceil(dif/2), w, c))
        down = np.zeros((floor(dif/2), w, c))
        new_img = np.concatenate((up, img, down), 0)
    return new_img

def reshape(cv2_img, size):
    img = cv2_img.copy()
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w, c = img.shape
    assert c in (3, 1)
    if h != w:
        img = pad_image(img)
    if (h, w) != size:
        img = resize_img(img, size)
    img = img.reshape(((c,) + size))
    img = img.astype("float32")
    if np.any(img > 1):
        img = img / 255
    return img

def contour_is_contained(cont, contours):
    x_1, y_1, w_1, h_1 = cv2.boundingRect(cont)
    for other_cont in contours:
        x_2, y_2, w_2, h_2 = cv2.boundingRect(other_cont)
        if x_1 > x_2 and y_1 > y_2 and w_1 < w_2 and h_1 < h_2:
            return True
    return False

def get_contours(img, morph=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
    if morph == "open":
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((2, 2), dtype="uint8"), iterations=1)
    elif morph == "close":
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((2, 2), dtype="uint8"), iterations=1)

    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours.sort(key=lambda c: cv2.boundingRect(c)[0])
    return thresh, contours

def draw_contours(thresh, contours):
    masks = []
    for cont in contours:
        if contour_is_contained(cont, contours):
            continue

        x, y, w, h = cv2.boundingRect(cont)

        mask = thresh[y:y+h, x:x+w]
        mask = np.pad(mask, 2, "constant", constant_values=(0,))
        masks.append(mask)
    return masks

def get_leftmost_coords(contour, img):
    min_x, min_y, w, h = cv2.boundingRect(contour)
    max_x = min_x + w
    max_y = min_y + h

    left_top = 0
    left_bot = 0
    for x in range(min_x, max_x, 1):
        if left_top == 0 and img[min_y+3, x] > 150:
            left_top = x
        if left_bot == 0 and img[max_y-3, x] > 150:
            left_bot = x

    return left_top, min_y, left_bot, max_y

def get_character_images(img):
    thresh, contours = get_contours(img)

    return draw_contours(thresh, contours)

def get_kda_digits(img):
    thresh, contours = get_contours(img, morph="open")

    masks = draw_contours(thresh.copy(), contours)
    filtered_masks = []
    digit_masks = []
    for index, mask in enumerate(masks):
        mid_y = mask.shape[0] // 2
        mid_x = mask.shape[1] // 2
        M = cv2.getRotationMatrix2D((mid_y, mid_x), 21, 1)
        rotated_img = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))
        new_contours = cv2.findContours(rotated_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
        new_mask = draw_contours(rotated_img, new_contours)[0]

        if new_mask.shape[1] > 10:
            digit_masks.append(mask)
        else:
            filtered_masks.append(digit_masks) # We have seen a "/" character.
            digit_masks = []

    filtered_masks.append(digit_masks)

    return filtered_masks

def get_cs_digits(img):
    thresh, contours = get_contours(img, morph="open")

    return draw_contours(thresh, contours)
