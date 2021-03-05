import json
from json.decoder import JSONDecodeError
import multiprocessing

def normalize(value, sum_values):
    return value / sum_values if sum_values > 0 else 0

def sparse_onehot_indices(data, lengths):
    len_summs, len_champs, len_items, total_length = lengths

    indices_x = []
    values = []

    for player_data, champ_data, item_data, summ_data, team_data in zip(*data):
        for stats, champ_id, items, summs in zip(player_data, champ_data, item_data, summ_data):
            # Add player stats (level, kills, deaths, assists, cs)
            for i in range(5):
                indices_x.append(i)
                values.append(stats[i])

            curr_x_index = 4

            # Add player summoner ids
            for summoner_spell_id in summs:
                indices_x.append(curr_x_index + summoner_spell_id)
                values.append(1)

            curr_x_index += len_summs + 1

            # Add player champion id
            indices_x.append(curr_x_index + champ_id)
            values.append(1)
        
            curr_x_index += len_champs

            # Add player item ids
            for item_id in items:
                indices_x.append(curr_x_index + item_id)
                values.append(1)

            curr_x_index += len_items

            # Add team stats (towers destoyed + dragons)
            for i in range(5):
                indices_x.append(curr_x_index + i)
                values.append(team_data[i])

    return (indices_x, values)

def structure_data_for_game(data_for_file):
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

        data = []

        # Insert data for each data frame.
        for frame in timeline_data["frames"]:
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

            total_kills = sum(kills)
            total_assists = sum(assists)
            total_deaths = sum(deaths)
            total_cs = sum(
                (
                    frame["participantFrames"][participantKey]["minionsKilled"] + 
                    frame["participantFrames"][participantKey]["jungleMinionsKilled"]
                )
                for participantKey in frame["participantFrames"]
            )
            total_towers = sum(towers)

            player_data = [[] for _ in range(2)]
            champ_data = [[] for _ in range(2)]
            summ_data = [[] for _ in range(2)]
            item_data = [[] for _ in range(2)]
            team_data = [[] for _ in range(2)]

            for participantKey in frame["participantFrames"]:
                participantData = frame["participantFrames"][participantKey]
                player_id = participantData["participantId"] - 1
                team_id = team_ids[player_id]
                cs = participantData["minionsKilled"] + participantData["jungleMinionsKilled"]

                champ_data[team_id].append(champion_ids[player_id])
                summ_data[team_id].append([
                    summoners_ids[player_id][i] for i in range(2)
                ])
                player_data[team_id].append([
                    normalize(participantData["level"], 18),
                    normalize(kills[player_id], total_kills),
                    normalize(deaths[player_id], total_deaths),
                    normalize(assists[player_id], total_assists),
                    normalize(cs, total_cs)
                ])

                if len(item_ids[player_id]) > 7:
                    if 34 in item_ids[player_id]:
                        item_ids[player_id].remove(34) # Remove Control Ward.
                    if 134 in item_ids[player_id]:
                        item_ids[player_id].remove(134) # Remove Oracle Lens.
                    if 133 in item_ids[player_id]:
                        item_ids[player_id].remove(133) # Remove Farsight Alteration.

                item_ids_list = list(item_ids[player_id])

                if len(item_ids_list) < 7:
                    item_ids_list.extend([0 for _ in range(7 - len(item_ids_list))])

                item_data[team_id].append(item_ids_list)

            for team_id, towers_destroyed in enumerate(towers):
                normed_dragons = [
                    normalize(count, 4) for count in dragons[team_id]
                ]
                team_data[team_id] = (
                    [normalize(towers_destroyed,total_towers)] + normed_dragons
                )

            len_matrix = len(summ_indices) + len(champ_indices) + len(item_indices) + 10
            lengths = len(summ_indices), len(champ_indices), len(item_indices), len_matrix
            matrix_data = (player_data, champ_data, item_data, summ_data, team_data)
            sparse_data = sparse_onehot_indices(matrix_data, lengths)

            data.append(sparse_data)

        team_data = match_data["teams"]
        blue_won = int(not ((team_data[0]["teamId"] == 100) ^ (team_data[0]["win"] == "Win")))

        game_id = match_data["gameId"]
        len_frames = len(data)
        filename = f"data/training_data/labeled_games/{game_id}_{len_frames}.csv"

        with open(filename, "w", encoding="utf-8") as fp:
            for indices_x, values in data:
                indices_str = ",".join(str(i) for i in indices_x)
                values_str = ",".join(f"{v:.0f}" if v < 0.00001 or v > 0.99999 else f"{v:.4f}" for v in values)
                fp.write(indices_str + "\n")
                fp.write(values_str + "\n")
            fp.write(f"{blue_won}")
    except JSONDecodeError as exc:
        print(match_file)
        raise exc

def pool_structuring_task(games):
    for game_data in games:
        structure_data_for_game(game_data)

def structure_data_in_parallel(game_files, load_processes=8):
    with multiprocessing.Pool(processes=load_processes) as pool:
        async_results = []
        split_ratio = len(game_files) // load_processes
        for task_index in range(load_processes):
            split_data = game_files[task_index * split_ratio: (task_index + 1) * split_ratio]
            async_results.append(
                pool.apply_async(
                    pool_structuring_task,
                    (split_data,)
                )
            )

        for result in async_results:
            result.get()

def load_data_for_game(game_file):
    with open(game_file, "r", encoding="utf-8") as fp:
        data = fp.readlines()
        frame_indices = []
        frame_values = []
        for data_index in range(0, len(data)-1, 2):
            indices = [int(x) for x in data[data_index].split(",")]
            values = [float(x) for x in data[data_index+1].split(",")]
            if len(values) == 200:
                frame_indices.append(indices)
                frame_values.append(values)
            else:
                print(f"Skipped erroneous data point in file '{game_file}'.", flush=True)
        label = float(data[-1])

        return (frame_indices, frame_values, label)

def pool_loading_task(games):
    data = []
    for game_data in games:
        frame_indices, frame_values, frame_label = load_data_for_game(game_data)
        data.extend(zip(frame_indices, frame_values, (frame_label for _ in frame_values)))
    return data

def load_data_in_parallel(game_files, load_processes=32):
    with multiprocessing.Pool(processes=load_processes) as pool:
        async_results = []
        split_ratio = len(game_files) // load_processes
        for task_index in range(load_processes):
            split_data = game_files[task_index * split_ratio: (task_index + 1) * split_ratio]
            async_results.append(
                pool.apply_async(
                    pool_loading_task, (split_data,)
                )
            )

        all_data = []

        for result in async_results:
            process_data = result.get()
            all_data.extend(process_data)

        return all_data
