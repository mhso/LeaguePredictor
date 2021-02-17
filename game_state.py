from time import time
import cv2
import numpy as np
import win32gui
from game_data import ChampionData

TEST = True

def get_champ_portraits(img):
    # Left top champ: 510, 346
    # Right top champ: 1085, 346
    x_coords = [510, 1084]
    y_start = 350
    y_step = 76
    portrait_h = 50
    portrait_w = 50
    portraits = []
    for x in x_coords:
        for y_index in range(5):
            y = y_start + y_index * y_step
            y_1 = y - (portrait_h // 2)
            y_2 = y + (portrait_h // 2)
            x_1 = x - (portrait_w // 2)
            x_2 = x + (portrait_w // 2)
            portrait_img = img[y_1:y_2, x_1:x_2, :]
            portraits.append(portrait_img)
    return portraits

def get_champ_levels(portraits):
    x = 38
    y = 36
    level_images = []
    for portrait in portraits:
        level_img = portrait[y:y+10, x:x+14, :]
        resized_shape = tuple(x * 2 for x in level_img.shape[:2])
        resized = cv2.resize(level_img, resized_shape, interpolation=cv2.INTER_CUBIC)
        level_images.append(resized)
    return level_images

def get_similarities_hist(champ_portrait, all_portraits):
    similarities = []
    gray_captured = cv2.cvtColor(champ_portrait, cv2.COLOR_BGR2GRAY)
    histogram_captured = cv2.calcHist(
        [gray_captured], [0], None, [256], [0, 256]
    )
    for real_portrait in all_portraits:
        gray_real = cv2.cvtColor(real_portrait, cv2.COLOR_BGR2GRAY)
        histogram_real = cv2.calcHist(
            [gray_real], [0], None, [256], [0, 256]
        )
        i = 0
        diff = 0
        while i < len(histogram_captured) and i < len(histogram_real): 
            diff += (histogram_captured[i] - histogram_real[i]) ** 2
            i += 1
        diff = diff ** (1 / 2)
        similarities.append(diff[0])

    max_diff = max(similarities)
    normed_diff = [x / max_diff for x in similarities]
    return normed_diff

def reshape_portrait_image(portrait, size):
    size_offset = 4
    smaller_size = tuple(s - size_offset for s in size)
    resized = cv2.resize(portrait, smaller_size, interpolation=cv2.INTER_CUBIC)
    pads = [(5, 5), (3, 5), (0, 0)]
    resized = np.pad(resized, pads, "constant", constant_values=(0, 0))
    return resized

def mask_portrait(portrait):
    center = portrait.shape[0] // 2, portrait.shape[1] // 2
    radius = portrait.shape[0] // 2 + 12
    cv2.circle(portrait, center, radius, (0, 0, 0), 30)
    return portrait

def get_similarities(champ_portrait, actual_portraits):
    similarities = []

    pads = [(1, 0), (0, 0), (0, 0)]
    champ_portrait = np.pad(champ_portrait, pads, "constant", constant_values=(0, 0))

    masked_captured = mask_portrait(champ_portrait)

    for real_portrait in actual_portraits:
        masked_real = mask_portrait(reshape_portrait_image(real_portrait, masked_captured.shape[:2]))
        similarity = cv2.matchTemplate(masked_captured, masked_real, cv2.TM_CCOEFF_NORMED)
        mean = np.mean(similarity)

        similarities.append(mean)

    return similarities

class GameState:
    GAME_IN_PROGRESS = 1
    GAME_OVER = 0
    GAME_NOT_STARTED = -1

    def __init__(self, champ_data):
        self.champ_data = champ_data
        self.active_game_data = None
        self.last_api_call = 0
        self.api_call_interval = 30

    def get_active_champs(self):
        champ_names = []
        for participant in self.active_game_data["participants"]:
            champ_names.append(self.champ_data.get_champ_name(participant["championId"]))
        return champ_names

    def get_champ_names(self, portraits):
        active_champs = self.get_active_champs()
        actual_portraits = [
            self.champ_data.get_champion_portrait(champ_name)
            for champ_name in active_champs
        ]
        similarities = []
        for captured_portrait in portraits:
            similarity = get_similarities(captured_portrait, actual_portraits)
            similarities.append(similarity)

        best_matches = []
        for champ_similarities in similarities:
            best_match = 0
            max_similarity = -1
            for index, similarity in enumerate(champ_similarities):
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = index

            best_matches.append(best_match)

        for champ_index, best_match in enumerate(best_matches):
            champ_match = active_champs[best_match]
            print(f"Champion {champ_index+1} is most likely {champ_match}")

    def get_game_state(self, screen_grab):
        active_game_data = self.active_game_data
        if time() - self.last_api_call > self.api_call_interval:
            active_game_data = self.champ_data.get_active_game_data()

        if active_game_data is None:
            if self.active_game_data is not None:
                self.active_game_data = None
                return (GameState.GAME_OVER, None) # Game is now over.
            else:
                return (GameState.GAME_NOT_STARTED, None)

        self.active_game_data = active_game_data

        portraits = get_champ_portraits(screen_grab)
        level_images = get_champ_levels(portraits)
        # TODO: Get actual numbers from OCR classifier...

        champ_names = self.get_champ_names(portraits)

        return (GameState.GAME_IN_PROGRESS, champ_names)

class TestChampData(ChampionData):
    def get_active_game_data(self):
        return {
            "participants": [
                {"championId": 85}, # Kennen
                {"championId": 234}, # Viego
                {"championId": 777}, # Yone
                {"championId": 21}, # MF
                {"championId": 412}, # Thresh
                {"championId": 875}, # Sett
                {"championId": 102}, # Shyvana
                {"championId": 38}, # Kassadin
                {"championId": 360}, # Samira
                {"championId": 53} # Blitz
            ]
        }

if __name__ == "__main__":
    img = cv2.imread("test_data/frame_1.png", cv2.IMREAD_COLOR)
    champion_data = TestChampData()
    gui_state = GameState(champion_data)
    state = gui_state.get_game_state(img)
    print(state)
