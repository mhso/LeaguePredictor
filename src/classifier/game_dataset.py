from glob import glob
from time import time
import multiprocessing
import json
from json.decoder import JSONDecodeError
import numpy as np
if __name__ == "__main__":
    import torch
import config
from game_data import GameData

def group_data(files, game_data_handler):
    data_dict = {}

    for filename in files:
        name_split = filename.replace("\\", "/").split("/")[-1].split(".")[0].split("_")
        match_id = int(name_split[0])
        type = name_split[1]
        if match_id not in data_dict:
            data_dict[match_id] = []

        data_dict[match_id].append((type, filename))

    champ_indices = game_data_handler.champ_index
    item_indices = game_data_handler.item_index
    summ_indices = game_data_handler.summ_index

    grouped_data = []
    for match_id in data_dict:
        if len(data_dict[match_id]) == 2:
            type_1, datapoint_1 = data_dict[match_id][0]
            _, datapoint_2 = data_dict[match_id][1]

            if type_1 == "match":
                tup = (datapoint_1, datapoint_2)
            else:
                tup = (datapoint_2, datapoint_1)

            tup = tup + (champ_indices, item_indices, summ_indices)

            grouped_data.append(tup)

    return grouped_data

def shape_input(data):
    return data

def get_data_from_file(data_for_file):
    (match_file, timeline_file,
     champ_indices, item_indices, summ_indices) = data_for_file
    try:
        with open(match_file, encoding="utf-8") as match_fp:
            match_data = json.load(match_fp)
        with open(timeline_file, encoding="utf-8") as timeline_fp:
            timeline_data = json.load(timeline_fp)

        if match_data["gameDuration"] < 60 * 5:
            return None # The match was a remake.

        team_ids = [0 for _ in range(10)]
        item_ids = [set() for _ in range(10)]
        kills = [0 for _ in range(10)]
        assists = [0 for _ in range(10)]
        deaths = [0 for _ in range(10)]
        deaths = [0 for _ in range(10)]
        towers = [0 for _ in range(2)]
        dragon_ids = ["FIRE_DRAGON", "WATER_DRAGON", "EARTH_DRAGON", "AIR_DRAGON"]
        dragons = [[0, 0, 0, 0] for _ in range(2)]
        champion_ids = [0 for _ in range(10)]
        summoners_ids = [[0, 0] for _ in range(10)]

        for participantData in match_data["participants"]:
            player_id = participantData["participantId"] - 1
            team_ids[player_id] = 0 if participantData["teamId"] == 100 else 1
            champion_ids[player_id] = champ_indices[participantData["championId"]]

            summoners_ids[player_id][0] = participantData["spell1Id"]
            summoners_ids[player_id][1] = participantData["spell2Id"]
            # summoners_ids[player_id] = [
            #     summoners_ids[participantData["spell1Id"]], summoners_ids[participantData["spell2Id"]]
            # ]

        data = []

        # Insert data for each data frame.
        for frame in timeline_data["frames"]:
            data_frame = [[] for _ in range(2)]

            for eventData in frame["events"]:
                if eventData["type"] == "CHAMPION_KILL":
                    killer_id = eventData["killerId"] - 1
                    dead_id = eventData["victimId"] - 1
                    assisting_ids = eventData["assistingParticipantIds"]
                    kills[killer_id] += 1
                    for assist_id in assisting_ids:
                        assists[assist_id - 1] += 1
                    deaths[dead_id] += 1
                elif eventData["type"] == "BUILDING_KILL":
                    killer_id = eventData["killerId"] - 1
                    team_id = team_ids[killer_id]
                    towers[team_id] += 1
                elif eventData["type"] == "ELITE_MONSTER_KILL" and eventData["monsterType"] == "DRAGON" and eventData["monsterSubType"] in dragon_ids:
                    dragon_type = eventData["monsterSubType"]
                    killer_id = eventData["killerId"] - 1
                    team_id = team_ids[killer_id]
                    dragon_id = dragon_ids.index(dragon_type)
                    dragons[team_id][dragon_id] += 1
                elif eventData["type"] == "ITEM_PURCHASED":
                    item_id = item_indices.get(eventData["itemId"], 0)
                    if item_id in (132, 133, 134):
                        for trinket_id in (132, 133, 134):
                            if trinket_id in item_ids[player_id]:
                                item_ids[player_id].remove(trinket_id)

                    player_id = eventData["participantId"] - 1
                    item_ids[player_id].add(item_id)
                elif eventData["type"] in ("ITEM_SOLD", "ITEM_DESTROYED"):
                    player_id = eventData["participantId"] - 1
                    item_id = item_indices.get(eventData["itemId"], 0)
                    if item_id in item_ids[player_id]:
                        item_ids[player_id].remove(item_id)
                elif eventData["type"] == "ITEM_UNDO":
                    player_id = eventData["participantId"] - 1
                    item_to_add = eventData["afterId"]
                    if item_to_add != 0:
                        item_ids[player_id].add(item_indices.get(item_to_add, 0))

                    item_to_remove = eventData["beforeId"]
                    if item_to_remove != 0:
                        item_id = item_indices[item_to_remove]
                        if item_id in item_ids[player_id]:
                            item_ids[player_id].remove(item_id)

            for participantKey in frame["participantFrames"]:
                participantData = frame["participantFrames"][participantKey]
                player_id = participantData["participantId"] - 1
                team_id = team_ids[player_id]
                cs = participantData["minionsKilled"] + participantData["jungleMinionsKilled"]

                player_data = []
                player_data.extend([
                    champion_ids[player_id], participantData["level"],
                    summoners_ids[player_id][0], summoners_ids[player_id][1],
                ])

                if len(item_ids[player_id]) > 7:
                    if 34 in item_ids[player_id]:
                        item_ids[player_id].remove(34) # Remove Control Ward.
                    if 134 in item_ids[player_id]:
                        item_ids[player_id].remove(134) # Remove Oracle Lens.
                    if 133 in item_ids[player_id]:
                        item_ids[player_id].remove(133) # Remove Farsight Alteration.

                player_data.extend(item_ids[player_id])
                if len(item_ids[player_id]) < 7:
                    player_data.extend([0 for _ in range(7 - len(item_ids[player_id]))])

                player_data.extend([
                    kills[player_id], assists[player_id], deaths[player_id], cs
                ])
                if len(player_data) != 15:
                    print(f"Invalid player data: {player_data}", flush=True)
                data_frame[team_id].append(player_data)
    
            for team_id, towers_destroyed in enumerate(towers):
                data_for_team = [towers_destroyed] + dragons[team_id] + [0 for _ in range(10)]
                if len(data_for_team) != 15:
                    print(f"Invalid team data: {data_for_team}", flush=True)
                data_frame[team_id].append(data_for_team)

            data.append(data_frame)

        team_data = match_data["teams"]
        blue_won = int(not ((team_data[0]["teamId"] == 100) ^ (team_data[0]["win"] == "Win")))

        return [data, blue_won]
    except JSONDecodeError as exc:
        print(match_file)
        raise exc

def get_data(batch_size, validation_split):
    game_data_handler = GameData()

    files = glob(f"data/training_data/games/*.json")

    config.log(f"Game files: {len(files)}")

    grouped_data = group_data(files, game_data_handler)

    config.log(f"Complete game files: {len(grouped_data)}")

    time_start = time()

    data_processes = 8
    with multiprocessing.Pool(processes=data_processes) as pool:
        config.log(f"Loading game data in parallel across {data_processes} processes...")
        data = pool.map(get_data_from_file, grouped_data)

    data_x = []
    data_y = []
    for game_data in data:
        if game_data is not None:
            data_x.extend(game_data[0])
            data_y.extend([game_data[1] for _ in range(len(game_data[0]))])

    config.log(f"Loaded {len(data_x)} datapoints in {time() - time_start:.2f} seconds.")

    assert len(data_x) == len(data_y)

    for team_data in data_x:
        assert len(game_data) == 2
        for player_data in team_data:
            assert len(player_data) == 6
            for type_data in player_data:
                assert len(type_data) == 15

    config.log(f"Data has been validated and is well formed.")

    from classifier.dataset import create_dataloaders_dict, split_data

    config.log(f"Shuffling and splitting data...")
    x_train, y_train, x_test, y_test = split_data(data_x, data_y, validation_split)

    config.log(f"Creating data loaders...")

    return create_dataloaders_dict(batch_size, x_train, y_train, x_test, y_test)
