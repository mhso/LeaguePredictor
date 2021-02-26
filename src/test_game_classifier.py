from game_state import GameState, TestGameData
from classifier import game_dataset
from classifier.game_classifier import GameClassifier
from classifier.digit_classifier import DigitClassifier
from cv2 import imread, IMREAD_COLOR

if __name__ == "__main__":
    test_img_index = 8
    img = imread(f"test_data/frame_{test_img_index}.png", IMREAD_COLOR)
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

    pct = int(outcome * 100)

    print(f"Probability of win: {pct}%")
