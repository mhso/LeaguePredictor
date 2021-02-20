from time import sleep
from glob import glob
from PIL import ImageGrab
import cv2
import win32api
import numpy as np

TAB_KEY_HEX = 0x09
TAB_STATE = win32api.GetKeyState(TAB_KEY_HEX)

def tab_pressed():
    key_tab = win32api.GetKeyState(TAB_KEY_HEX)
    return key_tab != TAB_STATE and key_tab < 0

def format_number(num):
    to_str = str(num)
    if num < 1000:
        to_str = "0" + to_str
    if num < 100:
        to_str = "0" + to_str
    if num < 10:
        to_str = "0" + to_str
    return to_str

def get_latest_img():
    latest_img = glob("data/training_data/scoreboards/*.png")[-1]
    return int(latest_img.replace("\\", "/").split("/")[-1].split(".")[0])

def extract_digits():
    screenshot = ImageGrab.grab()
    cv2_img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    index = get_latest_img() + 1
    cv2.imwrite(f"data/training_data/scoreboards/{format_number(index)}.png", cv2_img)
    print("Captured image")

TAB_PRESSED = False

while True:
    if not TAB_PRESSED and tab_pressed():
        sleep(0.25)
        extract_digits()
        TAB_PRESSED = True
    elif TAB_PRESSED and not tab_pressed():
        TAB_PRESSED = False

    sleep(0.5)
