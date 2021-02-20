from glob import glob
from time import time
import multiprocessing
import json
from json.decoder import JSONDecodeError
import numpy as np
if __name__ == "__main__":
    import torch
    from classifier.dataset import Data, create_dataloaders_dict
import config
from game_data import GameData

def group_data(files):
    data_dict = {}

    for filename in files:
        name_split = filename.replace("\\", "/").split("/")[-1].split(".")[0].split("_")
        match_id = int(name_split[0])
        type = name_split[1]
        if match_id not in data_dict:
            data_dict[match_id] = []

        data_dict[match_id].append((type, filename))

    grouped_data = []
    for match_id in data_dict:
        if len(data_dict[match_id]) == 2:
            type_1, datapoint_1 = data_dict[match_id][0]
            _, datapoint_2 = data_dict[match_id][1]

            if type_1 == "match":
                tup = (datapoint_1, datapoint_2)
            else:
                tup = (datapoint_2, datapoint_1)

            grouped_data.append(tup)

    return grouped_data

def shape_input(data):
    return data

def get_data_from_file(filenames):
    data = []
    match_file, timeline_file = filenames
    try:
        with open(match_file, encoding="utf-8") as match_fp:
            match_data = json.load(match_fp)
        with open(timeline_file, encoding="utf-8") as timeline_fp:
            timeline_data = json.load(timeline_fp)

        data_blue = []
        data_red = []
        for participant in match_data["participants"]:
            if participant["teamId"] == 100:
                data_blue.append(participant["championId"])
            else:
                data_red.append(participant["championId"])

        team_data = match_data["teams"]
        blue_won = int(not ((team_data[0]["teamId"] == 100) ^ (team_data[0]["win"] == "Win")))

        data = [[data_blue, data_red], blue_won]

        return data
    except JSONDecodeError as exc:
        print(match_file)
        raise exc

def get_data(validation_split):
    game_data_handler = GameData()
    data_x = []
    data_y = []
    files = glob(f"data/training_data/games/*.json")

    config.log(f"Datapoints: {len(files)}")

    grouped_data = group_data(files)

    config.log(f"Complete datapoints: {len(grouped_data)}")

    time_start = time()

    data_processes = 8
    with multiprocessing.Pool(processes=data_processes) as pool:
        config.log(f"Loading game data in parallel across {data_processes} processes...")
        data = pool.map(get_data_from_file, grouped_data)

    config.log(f"Loaded {len(data)} datapoints in {time() - time_start:.2f} seconds.")

    champion_ids = game_data_handler.get_champion_ids()
    games_for_champ = {champ_id: 0 for champ_id in champion_ids}
    wins_for_champ = {champ_id: 0 for champ_id in champion_ids}

    for game_data in data:
        blue_champs = game_data[0][0]
        red_champs = game_data[0][1]
        blue_won = game_data[1]
        for champ_id in blue_champs:
            games_for_champ[champ_id] += 1
            if blue_won:
                wins_for_champ[champ_id] += 1
        for champ_id in red_champs:
            games_for_champ[champ_id] += 1
            if not blue_won:
                wins_for_champ[champ_id] += 1

    for champ_id in games_for_champ:
        champ_name = game_data_handler.get_champ_name(champ_id)
        games = games_for_champ[champ_id]
        wins = wins_for_champ[champ_id]
        pct = int((wins / games) * 100)
        print(f"{champ_name} - Games: {games} - Wins: {wins} ({pct}%)")

    # assert len(data_x) == len(data_y)

    # shuffle_seed = 2042
    # np.random.seed(shuffle_seed)
    # np.random.shuffle(data_x)
    # np.random.seed(shuffle_seed)
    # np.random.shuffle(data_y)

    # split_index = int(len(data_x) * validation_split)
    # x_train = np.array(data_x[:split_index], dtype="float32")
    # y_train = np.array(data_y[:split_index], dtype="int64")
    # x_test = np.array(data_x[split_index:], dtype="float32")
    # y_test = np.array(data_y[split_index:], dtype="int64")

    # return x_train, y_train, x_test, y_test
