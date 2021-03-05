from glob import glob
import cv2
from game import game_state

FILE_PATH = "data/training_data"

def format_number(num):
    to_str = str(num)
    if num < 1000:
        to_str = "0" + to_str
    if num < 100:
        to_str = "0" + to_str
    if num < 10:
        to_str = "0" + to_str
    return to_str

def generate_digit_images(scoreboard_img):
    champ_portraits = game_state.get_champ_portraits(scoreboard_img)
    level_images = game_state.get_champ_level_images(champ_portraits)
    tower_images = game_state.get_towers_destroyed_images(scoreboard_img)
    player_scores_images = game_state.get_player_score_images(scoreboard_img)
    player_cs_images = game_state.get_player_cs_images(scoreboard_img)

    image_lists = [level_images, player_scores_images, player_cs_images]
    index = len(glob(f"{FILE_PATH}/digits/*.png")) + 1
    index_before = index
    for images in image_lists:
        for player_images in images:
            for mask_image in player_images:
                cv2.imwrite(f"{FILE_PATH}/digits/img_{format_number(index)}.png", mask_image)
                index += 1
    for team_images in tower_images:
        for mask_image in team_images:
            cv2.imwrite(f"{FILE_PATH}/digits/img_{format_number(index)}.png", mask_image)
            index += 1

    return index - index_before

scoreboard_images = glob(f"{FILE_PATH}/scoreboards/*.png")
images_processed = int(open(f"{FILE_PATH}/scoreboards/info.txt").readline().split("=")[1])

total_images = 0
for filename in scoreboard_images[images_processed:]:
    img = cv2.imread(filename, cv2.IMREAD_COLOR)
    total_images += generate_digit_images(img)
    images_processed += 1

print(f"Generated {total_images} images.")

with open(f"{FILE_PATH}/scoreboards/info.txt", "w") as fp:
    fp.write(f"processed={images_processed}")
