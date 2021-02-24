from glob import glob
from time import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from classifier.game_dataset_util import load_data_in_parallel, sparse_onehot_indices
import config

DATA_PATH = "data/training_data/labeled_games"

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
        self.files = []
        self.indices = {}
        self.x = []
        self.y = []
        self.x_index = 0
        self.data = load_data_in_parallel(game_files, load_processes)

        self.size = len(self.data)#data_index + 1

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

def split_data(files, validation_split, seed=2042):
    np.random.seed(seed)
    np.random.shuffle(files)

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
        for index, player_data in enumerate(data[team_key]["players"]):
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

        for index, data_for_player in enumerate(data[team_key]["players"]):
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

    # assert len(reshaped) == 2
    # for player_data in reshaped:
    #     assert len(player_data) == 6
    #     for type_data in player_data:
    #         assert len(type_data) == 15

    return reshaped.to_dense().view([1] + size)

def get_data(batch_size, validation_split):
    files = glob(f"{DATA_PATH}/*.csv")

    config.log(f"Game files: {len(files)}")

    time_start = time()

    config.log(f"Shuffling and splitting data...")
    time_start = time()

    files_train, files_test = split_data(files, validation_split)

    duration = time() - time_start
    config.log(f"Done in {duration:.2f} seconds.")

    config.log(f"Creating data loaders...")

    return create_dataloaders_dict(batch_size, files_train, files_test)
