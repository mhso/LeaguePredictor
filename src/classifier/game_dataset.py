from glob import glob
import numpy as np
import torch
from game.game_data import GameData
from torch.utils.data import Dataset, DataLoader
from classifier.game_dataset_util import load_data_in_parallel, sparse_onehot_indices
import config

def visualize_data_matrix(datapoint):
    game_data = GameData()
    sparse_tensor, label = datapoint
    dense = sparse_tensor.to_dense()

    team_names = ["Blue", "Red"]

    for team_id in range(2):
        print(f"===== {team_names[team_id]} team =====")
        for player_id in range(5):
            player_data = dense[team_id][player_id]
            level = player_data[0] * 18
            kills_ratio = player_data[1]
            deaths_ratio = player_data[2]
            assists_ratio = player_data[3]
            cs_ratio = player_data[4]
            all_champ_names = game_data.get_champion_names()
            all_item_ids = game_data.get_item_ids()
            summ_start = 5
            num_summs = len(game_data.summ_index)
            summ_names = []
            for summ_index in range(summ_start, summ_start+num_summs):
                if player_data[summ_index] == 1:
                    summ_names.append(game_data.get_summoner_spell_name(summ_index-summ_start+1))

            champ_start = summ_start+num_summs
            num_champs = len(game_data.champ_index)
            champ_name = None
            for champ_index in range(champ_start, champ_start+num_champs):
                if player_data[champ_index] == 1:
                    champ_name = all_champ_names[champ_index-champ_start]
                    break

            item_start = champ_start+num_champs
            num_items = len(game_data.item_index)
            item_names = []
            for item_index in range(item_start, item_start+num_items):
                if player_data[item_index] == 1:
                    item_id = all_item_ids[item_index-item_start]
                    item_names.append(game_data.get_item_name(item_id))

            player_str = (
                f"- Player {player_id+1}: Level: {level}, Kills: {kills_ratio}, "
                f"Assists: {assists_ratio}, Deaths: {deaths_ratio}, CS: {cs_ratio}, "
                f"Champ: {champ_name}, Summoners: {summ_names}, Items: {item_names}"
            )
            print(player_str)
        towers_ratio = dense[team_id][0][-5]
        dragons = dense[team_id][0][-4:]
        print(f"Towers: {towers_ratio}, Dragons: {dragons}")

    print(f"BLUE WON: {bool(label)}")

def create_sparse_tensor(indices_x, values, size):
    indices_z = []
    indices_y = []
    for team_index in range(2):
        for player_index in range(5):
            indices_z.extend([team_index for _ in range(len(values) // 10)])
            indices_y.extend([player_index for _ in range(len(values) // 10)])                

    sparse_tensor = torch.sparse.FloatTensor(
        torch.LongTensor([indices_z, indices_y, indices_x]),
        torch.FloatTensor(values),
        torch.Size(size)
    )
    return sparse_tensor

class Data(Dataset):
    def __init__(self, game_files, load_processes):
        # Load the structured data in parallel (this is very memory intensive).
        self.data = load_data_in_parallel(game_files, load_processes)

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        data_entry = self.data[index]
        size = list(config.GAME_INPUT_DIM)
        return create_sparse_tensor(data_entry[0], data_entry[1], size), data_entry[2]

def create_dataloaders_dict(batch_size, files_train, files_test):
    datasets = {}
    load_processes = 32
    config.log(f"Loading training data in parallel across {load_processes} processes...")
    datasets['train'] = Data(files_train, load_processes)
    config.log(f"Loading validation data in parallel across {load_processes} processes...")
    datasets['val'] = Data(files_test, load_processes)
    dataloaders = {
        x: DataLoader(
            datasets[x],
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        ) for x in ('train', 'val')
    }
    return dataloaders

def split_data(files, validation_split):
    split_index = int(len(files) * validation_split)
    files_train = files[:split_index]
    files_test = files[split_index:]

    return files_train, files_test

def normalize(value, total):
    return value / total if total > 0 else 0

def shape_input(data, game_data_handler):
    total_kills = 0
    total_deaths = 0
    total_assists = 0
    total_towers = 0
    total_cs = 0
    num_champs = len(game_data_handler.champ_index)
    num_items = len(game_data_handler.item_index)
    num_summs = len(game_data_handler.summ_index)

    for team_key in ("blue", "red"):
        # Calculate total values for every relevant stat.
        for player_data in data[team_key]["players"]:
            total_kills += player_data["kills"]
            total_deaths += player_data["deaths"]
            total_assists += player_data["assists"]
            total_cs += player_data["cs"]

        total_towers += data[team_key]["towers_destroyed"]

    player_data = [[] for _ in range(2)]
    champ_data = [[] for _ in range(2)]
    summ_data = [[] for _ in range(2)]
    item_data = [[] for _ in range(2)]
    team_data = [[] for _ in range(2)]

    reshaped = []
    for team_key in ("blue", "red"):
        team_id = 0 if team_key == "blue" else 1

        for data_for_player in data[team_key]["players"]:
            player_data[team_id].append([
                normalize(data_for_player["level"], 18),
                normalize(data_for_player["kills"], total_kills),
                normalize(data_for_player["deaths"], total_deaths),
                normalize(data_for_player["assists"], total_assists),
                normalize(data_for_player["cs"], total_cs)
            ])
            champ_data[team_id].append(data_for_player["champ_id"])
            summ_data[team_id].append([
                data_for_player["summ_spell_id_1"], data_for_player["summ_spell_id_2"]
            ])

            item_data[team_id].append(data_for_player["item_ids"])

        normed_dragons = [
            normalize(count, 4) for count in data[team_key]["dragons"]
        ]
        team_data[team_id] = (
            [normalize(data[team_key]["towers_destroyed"], total_towers)] + normed_dragons
        )

    len_matrix = num_summs + num_champs + num_items + 10
    lengths = num_summs, num_champs, num_items, len_matrix
    matrix_data = (player_data, champ_data, item_data, summ_data, team_data)
    indices_x, values = sparse_onehot_indices(matrix_data, lengths)
    size = list(config.GAME_INPUT_DIM)

    reshaped = create_sparse_tensor(indices_x, values, size)

    return reshaped.to_dense().view([1] + size)

def get_data_files():
    data_path = "data/training_data/labeled_games"
    files = glob(f"{data_path}/*.csv")
    config.log(f"Game files: {len(files)}")

    seed = 2042
    np.random.seed(seed)
    np.random.shuffle(files)
    return files

def get_data(chunk_start, chunk_end, game_files, validation_split):
    if chunk_end >= len(game_files):
        chunk_end = len(game_files)
    files_in_chunk = game_files[chunk_start:chunk_end]
    files_train, files_test = split_data(files_in_chunk, validation_split)

    return files_train, files_test
