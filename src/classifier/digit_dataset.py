from glob import glob
import cv2
import numpy as np
import torch
import config
from classifier.dataset import Data, create_dataloaders_dict

def shape_input(images):
    if not isinstance(images, list):
        images = [images]

    return torch.tensor(
        [reshape_img(img) for img in images]
    ).float()

def reshape_img(image):
    size = config.CHAR_INPUT_DIM[1:]
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    largest_dim_size = max(image.shape[:2])
    largest_dim = 0 if image.shape[0] == largest_dim_size else 1
    smallest_dim = abs(largest_dim - 1)
    ratio = size[largest_dim] / largest_dim_size
    other_size = int(image.shape[smallest_dim] * ratio)
    desired_size = [0, 0]
    desired_size[smallest_dim] = size[0] # Switch values cause y = x and x = y in numpy.
    desired_size[largest_dim] = other_size

    resized = cv2.resize(image, tuple(desired_size), interpolation=cv2.INTER_LINEAR)
    padding = [(0, 0), (0, 0), (0, 0)]
    size_diff = size[0] - other_size
    offset = 1 if size_diff % 2 == 1 else 0
    padding[smallest_dim] = (size_diff // 2, size_diff // 2 + offset)
    padded = np.pad(resized, padding, "constant", constant_values=(0,))

    reshaped = padded.reshape(config.CHAR_INPUT_DIM).astype("float32")
    if np.any(reshaped > 1):
        reshaped = reshaped / 255
    return reshaped

def get_data(validation_split):
    data_x = []
    data_y = []
    for digit in range(10):
        image_files = glob(f"data/training_data/labeled_digits/{digit}/*.png")
        for filename in image_files:
            image = cv2.imread(filename, cv2.IMREAD_COLOR)
            reshaped = reshape_img(image)

            data_x.append(reshaped)
            data_y.append(digit)

    assert len(data_x) == len(data_y)

    shuffle_seed = 2042
    np.random.seed(shuffle_seed)
    np.random.shuffle(data_x)
    np.random.seed(shuffle_seed)
    np.random.shuffle(data_y)

    split_index = int(len(data_x) * validation_split)
    x_train = np.array(data_x[:split_index], dtype="float32")
    y_train = np.array(data_y[:split_index], dtype="int64")
    x_test = np.array(data_x[split_index:], dtype="float32")
    y_test = np.array(data_y[split_index:], dtype="int64")

    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    data = get_data(0.7)
    x_train, y_train, x_test, y_test = data
    for x, y in zip(x_train, y_train):
        reshaped = x.reshape((x.shape[1], x.shape[2], 3))
        print(f"Label: {y}", flush=True)
        cv2.imshow("Digit", reshaped)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
