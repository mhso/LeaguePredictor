from torch.utils.data import Dataset, DataLoader
from torch import tensor
import numpy as np

class Data(Dataset):
    def __init__(self, x, y):
        self.x = [tensor(e).float() for e in x]
        self.y = y
        self.size = len(x)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.x[index], self.y[index]

def create_dataloaders_dict(batch_size, x_train, y_train, x_test, y_test):
    datasets = {}
    datasets['train'] = Data(x_train, y_train)
    datasets['val'] = Data(x_test, y_test)
    dataloaders = {
        x: DataLoader(
            datasets[x],
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        ) for x in ('train', 'val')
    }
    return dataloaders

def split_data(data_x, data_y, validation_split, seed=2042):
    np.random.seed(seed)
    np.random.shuffle(data_x)
    np.random.seed(seed)
    np.random.shuffle(data_y)

    split_index = int(len(data_x) * validation_split)
    x_train = np.array(data_x[:split_index], dtype="float32")
    y_train = np.array(data_y[:split_index], dtype="int64")
    x_test = np.array(data_x[split_index:], dtype="float32")
    y_test = np.array(data_y[split_index:], dtype="int64")

    return x_train, y_train, x_test, y_test
