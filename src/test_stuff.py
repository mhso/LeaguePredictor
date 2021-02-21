from glob import glob
from classifier import game_dataset

if __name__ == "__main__":
    game_dataset.get_data(64, 0.7)
