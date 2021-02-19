from torch.utils.data import Dataset, DataLoader
from torch import tensor

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
