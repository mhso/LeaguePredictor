from glob import glob
from time import time
from classifier.game_dataset_util import structure_data_in_parallel
from game_data import GameData
import config

def group_files(files, game_data_handler, processed_checkpoint):
    data_dict = {}

    for filename in files:
        name_split = filename.replace("\\", "/").split("/")[-1].split(".")[0].split("_")
        match_id = int(name_split[0])
        if match_id > processed_checkpoint:
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

if __name__ == "__main__":
    game_data_handler = GameData()

    files = glob(f"data/training_data/games/*.json")

    config.log(f"Game files: {len(files)}")

    processed_checkpoint = int(
        open("data/training_data/games/info.txt", "r").readline().split("=")[1]
    )

    grouped_data = group_files(files, game_data_handler, processed_checkpoint)

    config.log(f"Games not processed: {len(grouped_data)}")

    time_start = time()

    load_processes = 32
    config.log(f"Loading and structuring game data in parallel across {load_processes} processes...")
    data_generator = structure_data_in_parallel(grouped_data, load_processes)

    duration = time() - time_start
    config.log(f"Done in {duration:.2f} seconds.")

    with open("data/training_data/games/info.txt", "w") as fp:
        last_game_id = int(files[-1].replace("\\", "/").split("/")[-1].split(".")[0].split("_")[0])
        fp.write(f"Processed to={last_game_id}")
