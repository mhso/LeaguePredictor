import numpy as np

def split_data(data_x, data_y, validation_split, seed=2042):
    np.random.seed(seed)
    np.random.shuffle(data_x)
    np.random.seed(seed)
    np.random.shuffle(data_y)

    split_index = int(len(data_x) * validation_split)
    x_train = data_x[:split_index]
    y_train = data_y[:split_index]
    x_test = data_x[split_index:]
    y_test = data_y[split_index:]

    return x_train, y_train, x_test, y_test
