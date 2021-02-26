from time import time
import cv2
import numpy as np
from win32 import win32gui
from game_data import GameData, MY_SUMM_ID
from classifier.digit_classifier import DigitClassifier
from features import feature_extraction
from classifier import digit_dataset, game_dataset
import config

TEST = True

def get_champ_portraits(img):
    # Left top champ: 510, 346
    # Right top champ: 1085, 346
    x_coords = [511, 1085]
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

def get_item_icons(img):
    # 714, 332
    # 715, 408
    # 715, 484

    # 715 -> 749 -> 783
    # 1286 -> 1320 -> 1358
    x_coords = [714, 1286]
    x_step = 34
    y_start = 332
    y_step = 76
    icon_h = 32
    icon_w = 32
    icons = []
    for x_start in x_coords:
        for y_index in range(5):
            icons_for_champ = []
            y = y_start + y_index * y_step
            for x_index in range(7):
                x = x_start + x_index * x_step
                y_2 = y + icon_h
                x_2 = x + icon_w
                icon_img = img[y:y_2, x:x_2, :]
                icons_for_champ.append(icon_img)
            icons.append(icons_for_champ)
    return icons

def get_champ_level_images(portraits):
    x = 38
    y = 36
    w = 12
    h = 12
    level_images = []
    for portrait in portraits:
        img = portrait[y:y+h, x:x+w, :]
        masked_imgs = feature_extraction.get_character_images(img)
        images_for_player = []
        for mask in masked_imgs:
            images_for_player.append(mask)
        level_images.append(images_for_player)
    return level_images

def get_average_colors(img):
    h, w = img.shape[:2]
    total_pixels = h * w
    acc_colors = [0, 0, 0]
    for y in range(h):
        for x in range(w):
            for c in range(3):
                acc_colors[c] += img[y, x, c]
    return tuple(x / total_pixels for x in acc_colors)

def determine_dragon(avg_r, avg_g, avg_b):
    if sum((avg_r, avg_g, avg_b)) < 100:
        return 0 # No dragon.

    search_values = [
        (107.815, 55.405, 26.7825), # Infernal
        (33.365, 84.8175, 71.55), # Ocean
        (94.915, 69.725, 50.865), # Mountain
        (86.005, 117.195, 128.87) # Cloud
    ]
    threshold = 15
    matches = []
    for index, (search_r, search_g, search_b) in enumerate(search_values, start=1):
        if (abs(avg_r - search_r) > threshold or abs(avg_g - search_g) > threshold
                or abs(avg_b - search_b) > threshold):
            continue
        matches.append(index)

    if len(matches) != 1:
        return 0 # We can't tell which dragon it is :(

    return matches[0]

def get_dragons(img):
    # 867, 196
    # 1039, 196 -> x_dist = 56
    x_coords = [867, 1039]
    x_step = 56
    y = 196
    w = 20
    h = 20
    dragons = []
    for x_side, x_start in enumerate(x_coords):
        delta_x = 1 if x_side == 1 else -1
        dragons_team = []
        for x_index in range(4):
            x = (x_start + (x_index * delta_x) * x_step)
            icon = img[y:y+h, x:x+w, :]
            avg_b, avg_g, avg_r = get_average_colors(icon)
            dragons_team.append(determine_dragon(avg_r, avg_g, avg_b))

        dragons.append(dragons_team)

    return dragons

def get_player_score_images(screen_grab):
    # 612, 334 -> 696, 334
    #  1186, 334 -> 1270, 334
    x_coords = [612, 1186]
    w = 84
    h = 20
    y_start = 334
    y_step = 76
    images = []
    for x in x_coords:
        for y_index in range(5):
            y = y_start + y_index * y_step
            img = screen_grab[y:y+h, x:x+w]
            digit_masks = feature_extraction.get_kda_digits(img)
            images_for_player = []
            for masks in digit_masks:
                images_for_player.append(masks)

            images.append(images_for_player)

    return images

def get_player_cs_images(screen_grab):
    x_coords = [560, 1134]
    w = 40
    h = 20
    y_start = 334
    y_step = 76
    images = []
    for x in x_coords:
        for y_index in range(5):
            y = y_start + y_index * y_step
            img = screen_grab[y:y+h, x:x+w]
            masked_img = feature_extraction.get_cs_digits(img)
            images_for_player = []
            for img in masked_img:
                images_for_player.append(img)

            images.append(images_for_player)

    return images

def get_towers_destroyed_images(screen_grab):
    # 818, 262 -> 842
    # 1082, 262
    x_coords = [818, 1082]
    y = 262
    w = 24
    h = 16
    images = []
    for x in x_coords:
        img = screen_grab[y:y+h, x:x+w]
        images_for_team = []
        masked_img = feature_extraction.get_character_images(img)
        for img in masked_img:
            images_for_team.append(img)
        images.append(images_for_team)
    return images

def reshape_image(portrait, size):
    size_offset = 4
    smaller_size = tuple(s - size_offset for s in size)
    resized = cv2.resize(portrait, smaller_size, interpolation=cv2.INTER_CUBIC)
    pads = [(5, 5), (3, 5), (0, 0)]
    resized = np.pad(resized, pads, "constant", constant_values=(0, 0))
    return resized

def get_similarities_fancy(captured_img, actual_images, transform=None):
    similarities = []

    sift = cv2.SIFT_create()
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary

    kp1, des1 = sift.detectAndCompute(captured_img, None)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    for actual_img in actual_images:
        transformed = actual_img
        if transform is not None:
            transformed = transform(transformed, captured_img)

        kp2, des2 = sift.detectAndCompute(transformed, None)

        matches = flann.knnMatch(des1, des2, k=2)

        similarity = 0
        for m, n in matches:
            if m.distance < 0.7*n.distance:
                similarity += n.distance - m.distance
        similarity = similarity / len(matches)

        similarities.append(similarity)

    return similarities

def get_similarities(captured_img, actual_images, transform=None):
    similarities = []

    for actual_img in actual_images:
        transformed = actual_img
        if transform is not None:
            transformed = transform(transformed, captured_img)

        similarity = cv2.matchTemplate(captured_img, transformed, cv2.TM_SQDIFF_NORMED)
        mean = np.mean(similarity)

        similarities.append(mean)

    return similarities

def get_best_matches(similarities, search_items, minimize):
    best_matches = []
    for champ_similarities in similarities:
        best_match = 0
        best_similarity = max(champ_similarities) if minimize else min(champ_similarities)
        for index, similarity in enumerate(champ_similarities):
            if (minimize and similarity < best_similarity) or (not minimize and similarity > best_similarity):
                best_similarity = similarity
                best_match = index

        best_matches.append(search_items[best_match])
    return best_matches

class GameState:
    GAME_IN_PROGRESS = 1
    GAME_OVER = 0
    GAME_NOT_STARTED = -1

    def __init__(self, game_data, digit_classifier):
        self.game_data = game_data
        self.digit_classifier = digit_classifier
        self.active_game_data = None
        self.last_api_call = 0
        self.api_call_interval = 30

    def get_active_champs(self):
        champ_names = []
        for participant in self.active_game_data["participants"]:
            champ_names.append((
                participant["championId"],
                self.game_data.get_champ_handle(participant["championId"])
            ))
        return champ_names

    def get_player_champions(self, portraits):
        active_champs = self.get_active_champs()
        actual_portraits = [
            self.game_data.get_champion_portrait(champ_name)
            for _, champ_name in active_champs
        ]

        def transform(actual_img, captured_img):
            return cv2.cvtColor(
                reshape_image(actual_img, captured_img.shape[:2]),
                cv2.COLOR_BGR2GRAY
            )

        similarities = []
        for captured_portrait in portraits:
            similarity = get_similarities_fancy(
                cv2.cvtColor(captured_portrait, cv2.COLOR_BGR2GRAY),
                actual_portraits, transform
            )
            similarities.append(similarity)

        best_matches = get_best_matches(similarities, active_champs, False)

        possible_conflicts = {}
        seen_champs = set()
        for index, (champ_id, champ_name) in enumerate(best_matches):
            if champ_id not in possible_conflicts:
                possible_conflicts[champ_id] = []
            possible_conflicts[champ_id].append((index, similarities[index], champ_name))
            seen_champs.add(champ_id)

        missing_champ = None
        for champ_id, champ_name in active_champs:
            if champ_id not in seen_champs:
                missing_champ = champ_id, champ_name

        for champ_id in possible_conflicts:
            data = possible_conflicts[champ_id]
            if len(data) == 2:
                config.log(
                    (
                        f"CONFILICTS IN CHAMPIONS! {data[0][2]} present twice, " +
                        f"{missing_champ[1]} missing."
                    ),
                    config.LOG_WARNING
                )
                max_similarity = 0
                max_similarity_index = 0
                for index, similarities, champ_name in data:
                    if max(similarities) > max_similarity:
                        max_similarity = max(similarities)
                        max_similarity_index = index

                for index, similarities, champ_name in data:
                    if index == max_similarity_index:
                        best_matches[index] = (champ_id, champ_name)
                    else:
                        best_matches[index] = missing_champ
                config.log("Fixed champion conflicts!")
            elif len(data) > 2:
                config.log("Too many champion conflicts, can't fix!", config.LOG_ERROR)

        return best_matches

    def get_player_items(self, item_icons):
        item_ids = self.game_data.get_item_ids()
        actual_icons = [
            self.game_data.get_item_icon(item_id)
            for item_id in item_ids
        ]

        def transform(actual_img, captured_img):
            smaller_size = tuple(s for s in captured_img.shape[:2])
            return cv2.resize(actual_img, smaller_size, interpolation=cv2.INTER_CUBIC)

        similarities = []
        for player_icons in item_icons:
            for captured_icon in player_icons:
                similarity = get_similarities(
                    captured_icon, actual_icons, transform
                )
                similarities.append(similarity)

        best_matches = get_best_matches(similarities, item_ids, True)
        names = [
            self.game_data.get_item_name(item_id) for item_id in best_matches
        ]

        items_per_player = []
        for index in range(10):
            item_index = index * 7
            items_per_player.append((
                best_matches[item_index:item_index+7],
                names[item_index:item_index+7],
            ))

        return items_per_player

    def get_towers_destroyed(self, tower_images):
        team_towers = []
        for team_images in tower_images:
            classifier_input = digit_dataset.shape_input(team_images)
            towers = self.digit_classifier.predict(classifier_input)
            team_towers.append(int("".join(str(x) for x in towers)))
        return team_towers

    def get_player_scores(self, score_images):
        player_scores = []
        for player_images in score_images:
            kda = []
            for value_type in player_images:
                classifier_input = digit_dataset.shape_input(value_type)
                digits = self.digit_classifier.predict(classifier_input)
                kda.append(int("".join(str(x) for x in digits)))
            player_scores.append(tuple(kda))
        return player_scores

    def predict_digits(self, images):
        data = []
        for player_images in images:
            data_for_player = []
            for masks in player_images:
                classifier_input = digit_dataset.shape_input(masks)
                digits = self.digit_classifier.predict(classifier_input)
                data_for_player.append(int("".join(str(x) for x in digits)))

            data.append(data_for_player)

        return tuple(data)

    def get_game_state(self, screen_grab):
        active_game_data = self.active_game_data
        if time() - self.last_api_call > self.api_call_interval:
            active_game_data = self.game_data.get_active_game_data()

        if active_game_data is None:
            if self.active_game_data is not None:
                self.active_game_data = None
                return (GameState.GAME_OVER, None) # Game is now over.
            else:
                return (GameState.GAME_NOT_STARTED, None)

        self.active_game_data = active_game_data

        champ_portraits = get_champ_portraits(screen_grab)
        item_icons = get_item_icons(screen_grab)
        player_level_images = get_champ_level_images(champ_portraits)

        tower_images = get_towers_destroyed_images(screen_grab)
        towers_destroyed = self.get_towers_destroyed(tower_images)

        player_score_images = get_player_score_images(screen_grab)
        player_scores = self.get_player_scores(player_score_images)

        player_cs_images = get_player_cs_images(screen_grab)

        player_levels, player_cs = self.predict_digits(
            [player_level_images, player_cs_images]
        )

        champion_data = self.get_player_champions(champ_portraits)
        item_data = self.get_player_items(item_icons)

        dragons = get_dragons(screen_grab)

        my_team = "red"
        game_data = {
            "blue": {
                "players": [{} for _ in range(5)]
            },
            "red": {
                "players": [{} for _ in range(5)],
            }
        }

        team_index = {}

        for participant in self.active_game_data["participants"]:
            team_key = "blue" if participant["teamId"] == 100 else "red"
            player_index = 0
            for index, (champ_id, champ_name) in enumerate(champion_data):
                if int(participant["championId"]) == champ_id:
                    player_index = index
                    break

            if player_index > 4:
                team_index[team_key] = 1
            else:
                team_index[team_key] = 0

            k, d, a = player_scores[player_index]

            champ_index = self.game_data.get_champion_index(champion_data[player_index][0])
            item_indices = [
                self.game_data.get_item_index(item_id)
                for item_id in item_data[player_index][0]
            ]

            data_for_player = {
                "champ_id": champ_index,
                "champ_name": champion_data[player_index][1],
                "summ_spell_id_1": participant["spell1Id"],
                "summ_spell_id_2": participant["spell2Id"],
                "item_ids": item_indices,
                "item_names": item_data[player_index][1],
                "level": player_levels[player_index],
                "kills": k, "deaths": d, "assists": a,
                "cs": player_cs[player_index]
            }

            player_team_index = player_index if player_index < 5 else player_index - 5

            game_data[team_key]["players"][player_team_index] = data_for_player

            if participant["summonerId"] == MY_SUMM_ID:
                my_team = team_key

        for team_key in team_index:
            index = team_index[team_key]
            dragon_vector = [0, 0, 0, 0]
            for dragon_id in dragons[index]:
                if dragon_id > 0:
                    dragon_vector[dragon_id-1] += 1
            game_data[team_key]["dragons"] = dragon_vector
            game_data[team_key]["towers_destroyed"] = towers_destroyed[index]

        return (GameState.GAME_IN_PROGRESS, (game_data, my_team))

class TestGameData(GameData):
    def __init__(self, test_image_index):
        super().__init__()
        self.champions = []
        self.summs = [
            (12, 4), (4, 11), (4, 14), (4, 7), (4, 14),
            (4, 12), (4, 11), (4, 12), (3, 4), (14, 4)
        ]
        self.summ_ids = [f"Dude {index+1}" for index in range(10)]
        if test_image_index == 1:
            self.champions = [85, 234, 777, 21, 412, 875, 102, 38, 360, 53]
            self.summ_ids[2] = MY_SUMM_ID
        elif test_image_index == 4:
            self.champions = [122, 517, 142, 51, 412, 111, 104, 7, 360, 63]
            self.summ_ids[7] = MY_SUMM_ID
        elif test_image_index == 5:
            self.champions = [223, 141, 4, 360, 432, 98, 11, 236, 53, 61]
            self.summ_ids[3] = MY_SUMM_ID
        elif test_image_index == 7:
            self.champions = [106, 69, 54, 110, 350, 8, 63, 81, 104, 89]
            self.summs = [
                (4, 6), (4, 1), (14, 4), (4, 3), (4, 3),
                (21, 4), (21, 4), (14, 4), (4, 21), (14, 4)
            ]
            self.summ_ids[1] = MY_SUMM_ID
        elif test_image_index == 8:
            self.champions = [34, 23, 10, 21, 19, 223, 876, 62, 235, 86]
            self.summs = [
                (4, 21), (4, 3), (4, 3), (1, 4), (4, 6),
                (4, 6), (4, 7), (14, 4), (4, 6), (4, 3)
            ]
            self.summ_ids[6] = MY_SUMM_ID

    def get_active_game_data(self):
        return {
            "participants": [
                {
                    "championId": self.champions[0], "teamId": 100, "summonerId": self.summ_ids[0],
                    "spell1Id": self.get_summoner_spell_index(self.summs[0][0]), # Teleport
                    "spell2Id": self.get_summoner_spell_index(self.summs[0][1]) # Flash
                },
                {
                    "championId": self.champions[1], "teamId": 100, "summonerId": self.summ_ids[1],
                    "spell1Id": self.get_summoner_spell_index(self.summs[1][0]), # Flash
                    "spell2Id": self.get_summoner_spell_index(self.summs[1][1]) # Smite
                },
                {
                    "championId": self.champions[2], "teamId": 100, "summonerId": self.summ_ids[2],
                    "spell1Id": self.get_summoner_spell_index(self.summs[2][0]), # Flash
                    "spell2Id": self.get_summoner_spell_index(self.summs[2][1]) # Ignite
                },
                {
                    "championId": self.champions[3], "teamId": 100, "summonerId": self.summ_ids[3],
                    "spell1Id": self.get_summoner_spell_index(self.summs[3][0]), # Flash
                    "spell2Id": self.get_summoner_spell_index(self.summs[3][1]) # Heal
                },
                {
                    "championId": self.champions[4], "teamId": 100, "summonerId": self.summ_ids[4],
                    "spell1Id": self.get_summoner_spell_index(self.summs[4][0]), # Flash
                    "spell2Id": self.get_summoner_spell_index(self.summs[4][1]) # Ignite
                },
                {
                    "championId": self.champions[5], "teamId": 200, "summonerId": self.summ_ids[5],
                    "spell1Id": self.get_summoner_spell_index(self.summs[5][0]), # Flash
                    "spell2Id": self.get_summoner_spell_index(self.summs[5][1]) # Teleport
                },
                {
                    "championId": self.champions[6], "teamId": 200, "summonerId": self.summ_ids[6],
                    "spell1Id": self.get_summoner_spell_index(self.summs[6][0]), # Flash
                    "spell2Id": self.get_summoner_spell_index(self.summs[6][1]) # Smite
                },
                {
                    "championId": self.champions[7], "teamId": 200, "summonerId": self.summ_ids[7],
                    "spell1Id": self.get_summoner_spell_index(self.summs[7][0]), # Flash
                    "spell2Id": self.get_summoner_spell_index(self.summs[7][1]) # Teleport
                },
                {
                    "championId": self.champions[8], "teamId": 200, "summonerId": self.summ_ids[8],
                    "spell1Id": self.get_summoner_spell_index(self.summs[8][0]), # Exhaust
                    "spell2Id": self.get_summoner_spell_index(self.summs[8][1]) # Flash
                },
                {
                    "championId": self.champions[9], "teamId": 200, "summonerId": self.summ_ids[9],
                    "spell1Id": self.get_summoner_spell_index(self.summs[9][0]), # Ignite
                    "spell2Id": self.get_summoner_spell_index(self.summs[9][1]) # Flash
                } 
            ]
        }

if __name__ == "__main__":
    test_img_index = 4
    img = cv2.imread(f"test_data/frame_{test_img_index}.png", cv2.IMREAD_COLOR)
    champion_data = TestGameData(test_img_index)
    digit_classifier = DigitClassifier()
    digit_classifier.load()
    game_state = GameState(champion_data, digit_classifier)
    state, data = game_state.get_game_state(img)
    game_data, my_team = data
    for team in game_data:
        print(f"====== {team.upper()} TEAM ======")
        print(f"Towers destroyed: {game_data[team]['towers_destroyed']}")
        print(f"Dragons: {game_data[team]['dragons']}")
        for player_data in game_data[team]["players"]:
            print(player_data)
            print("***********************************************")
    
