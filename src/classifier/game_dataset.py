from glob import glob
import os
import multiprocessing
import json
from json.decoder import JSONDecodeError
import numpy as np
import torch
import config
from classifier.dataset import Data, create_dataloaders_dict

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

def get_match_data(files):
    data = []
    for match_file, timeline_file in files:
        try:
            match_data = json.load(open(match_file, encoding="utf-8"))
            timeline_data = json.load(open(timeline_file, encoding="utf-8"))
            data.append((match_data, timeline_data))
        except JSONDecodeError:
            print(match_file)
    return data

def get_data(validation_split):
    data_x = []
    data_y = []
    files = glob(f"data/training_data/games/*.json")

    grouped_data = group_data(files)

    print(f"Complete datapoints: {len(grouped_data)}")

    data = get_match_data(grouped_data)

    print(f"Complete datapoints: {len(data)}")

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
