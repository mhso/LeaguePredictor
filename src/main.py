import win32api
from time import sleep

TAB_KEY_HEX = 0x09
TAB_STATE = win32api.GetKeyState(TAB_KEY_HEX)

def tab_pressed():
    key_tab = win32api.GetKeyState(TAB_KEY_HEX)
    return key_tab != TAB_STATE and key_tab < 0

TAB_PRESSED = False

while True:
    if not TAB_PRESSED and tab_pressed():
        TAB_PRESSED = True
    elif TAB_PRESSED and not tab_pressed():
        TAB_PRESSED = False

    sleep(0.5)
