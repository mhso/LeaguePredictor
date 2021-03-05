from sys import argv
from os.path import exists
import json
from game.game_state import GameState
from test_game_state import TestGameData
from classifier import game_dataset
from classifier.game_classifier import GameClassifier
from classifier.digit_classifier import DigitClassifier
import cv2

if __name__ == "__main__":
    if len(argv) < 2:
        print("Provide an index")

    test_img_index = int(argv[1])
    img = cv2.imread(f"test_data/frame_{test_img_index}.png", cv2.IMREAD_COLOR)
    game_data_handler = TestGameData(test_img_index)
    game_classifier = GameClassifier()
    game_classifier.load()
    digit_classifier = DigitClassifier()
    digit_classifier.load()

    game_state = GameState(game_data_handler, digit_classifier)
    state, data = game_state.get_game_state(img)
    game_data, my_team = data
    classifier_input = game_dataset.shape_input(game_data, game_data_handler)

    outcome = game_classifier.predict(classifier_input)
    if my_team == "red":
        outcome = 1 - outcome

    pct = f"{outcome * 100:.2f}"

    print(f"Probability of win: {pct}%", flush=True)

    shape = (img.shape[1] // 2, img.shape[0] // 2)

    resized = cv2.resize(img, shape, interpolation=cv2.INTER_AREA)

    cv2.putText(
        resized, f"Probability of win: {pct}%",
        (10, resized.shape[0] - 80),
        cv2.FONT_HERSHEY_COMPLEX, 2, (30, 230, 20), thickness=2
    )

    cv2.imshow("Image", resized)
    cv2.waitKey(0)
