from time import sleep
from PIL import ImageGrab
import win32api
import cv2
from numpy import array
import config
from game_state import GameState
from game_data import GameData
from classifier.digit_classifier import DigitClassifier
from classifier.game_classifier import GameClassifier
from classifier.game_dataset import shape_input

TAB_KEY_HEX = 0x09
TAB_STATE = win32api.GetKeyState(TAB_KEY_HEX)

def tab_pressed():
    key_tab = win32api.GetKeyState(TAB_KEY_HEX)
    return key_tab != TAB_STATE and key_tab < 0

def get_game_state(game_state_handler):
    screenshot = ImageGrab.grab()
    cv2_img = cv2.cvtColor(array(screenshot), cv2.COLOR_RGB2BGR)
    return game_state_handler.get_game_state(cv2_img)

def get_prediction(game_classifier, game_data):
    classifier_input = shape_input(game_data)
    # prediction = game_classifier.predict(classifier_input)
    return classifier_input

TAB_PRESSED = False

game_data_handler = GameData()
game_classifier = GameClassifier().cpu()
game_classifier.load()
digit_classifier = DigitClassifier().cpu()
digit_classifier.load()
game_state_handler = GameState(game_data_handler, digit_classifier)

while True:
    if not TAB_PRESSED and tab_pressed():
        TAB_PRESSED = True
        state, game_data = get_game_state(game_state_handler)
        if state == game_state_handler.GAME_OVER:
            config.log("Game over. Shutting down...")
            break
        if state == game_state_handler.GAME_IN_PROGRESS:
            #prediction = get_prediction(game_classifier, game_data)
            for team in ("blue", "red"):
                config.log(f"====== {team.upper()} TEAM ======", flush=True)
                for player_data in game_data[team]["players"]:
                    config.log(player_data, flush=True)
                    config.log("***********************************************", flush=True)
            config.log(f"My team: {game_data['my_team']}", flush=True)
    elif TAB_PRESSED and not tab_pressed():
        TAB_PRESSED = False

    sleep(0.5)
