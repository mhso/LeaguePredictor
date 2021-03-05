import json
from time import sleep
from PIL import ImageGrab
import win32api
import cv2
from numpy import array
import config
from game.game_state import GameState, GameStateException
from game.game_data import GameData
from game import transponder
from classifier.digit_classifier import DigitClassifier
from classifier.game_classifier import GameClassifier
from classifier.game_dataset import shape_input

TAB_KEY_HEX = 0x09
TAB_STATE = win32api.GetKeyState(TAB_KEY_HEX)

ALT_KEY_HEX = 0x12
ALT_STATE = win32api.GetKeyState(ALT_KEY_HEX)

def tab_pressed():
    key_tab = win32api.GetKeyState(TAB_KEY_HEX)
    return key_tab != TAB_STATE and key_tab < 0

def alt_pressed():
    key_alt = win32api.GetKeyState(ALT_KEY_HEX)
    return key_alt != TAB_STATE and key_alt < 0

def get_game_state(game_state_handler):
    screenshot = ImageGrab.grab()
    cv2_img = cv2.cvtColor(array(screenshot), cv2.COLOR_RGB2BGR)
    return game_state_handler.get_game_state(cv2_img)

def get_prediction(game_classifier, game_data, game_data_handler, my_team):
    classifier_input = shape_input(game_data, game_data_handler)
    outcome = game_classifier.predict(classifier_input)
    if my_team == "red":
        outcome = 1 - outcome
    return outcome

TAB_PRESSED = False
ALT_PRESSED = False

game_data_handler = GameData()
game_classifier = GameClassifier().cpu()
game_classifier.load()
digit_classifier = DigitClassifier().cpu()
digit_classifier.load()
game_state_handler = GameState(game_data_handler, digit_classifier)

auth = json.load(open("data/auth.json", encoding="utf-8"))

while True:
    if not ALT_PRESSED and alt_pressed():
        ALT_PRESSED = True
    elif ALT_PRESSED and not alt_pressed():
        ALT_PRESSED = False

    if not TAB_PRESSED and tab_pressed():
        TAB_PRESSED = True
        if not ALT_PRESSED:
            try:
                state, data = get_game_state(game_state_handler)

                if state == game_state_handler.GAME_OVER:
                    config.log("Game over! Waiting for new game...")
                if state == game_state_handler.GAME_IN_PROGRESS:
                    game_data, my_team = data
                    prediction = get_prediction(game_classifier, game_data, game_data_handler, my_team)

                    pct_win = f"{prediction * 100:.2f}"
                    game_id = game_state_handler.active_game_data["gameId"]
                    game_duration = game_state_handler.get_game_duration()

                    transpoder_data = {
                        "secret": auth["discordToken"], "pct_win": pct_win,
                        "game_id": game_id, "game_duration": game_duration
                    }
                    transponder.send_status(transpoder_data)

                    config.log(f"Chance of win: {pct_win}%")
                else:
                    config.log("Game not started yet. Nothing to do...")
            except GameStateException as exc:
                config.log(f"Error during 'get_game_state': {str(exc)}")
    elif TAB_PRESSED and not tab_pressed():
        TAB_PRESSED = False

    sleep(0.5)
