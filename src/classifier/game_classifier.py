import torch
import config

CONV_FILTERS = 256
KERNEL_SIZE = 3

RHO = 0.9
EPS = 1e-6
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-5

BATCH_SIZE = 64#256
CHUNK_SIZE = 4500 # Number of game files to load into memory at once.
VALIDATION_SPLIT = 0.75

DENSE_FEATURES = 512

OUTPUT_DIM = 10

class GameClassifier(torch.nn.Module):
    def __init__(self):
        super(GameClassifier, self).__init__()

        self.model_path = "game_classifier.pt"

        self.conv_1 = torch.nn.Conv2d(
            config.GAME_INPUT_DIM[0], CONV_FILTERS, KERNEL_SIZE, stride=1, padding=1
        )
        self.batch_1 = torch.nn.BatchNorm2d(CONV_FILTERS)
        self.relu_1 = torch.nn.LeakyReLU()

        self.conv_2 = torch.nn.Conv2d(
            CONV_FILTERS, CONV_FILTERS * 2, KERNEL_SIZE, stride=1, padding=1
        )
        self.batch_2 = torch.nn.BatchNorm2d(CONV_FILTERS * 2)
        self.relu_2 = torch.nn.LeakyReLU()

        self.drop_1 = torch.nn.Dropout(0.25)
        self.pool_1 = torch.nn.MaxPool2d(2, padding=0)

        self.dense_features_1 = CONV_FILTERS * 2 * (config.GAME_INPUT_DIM[1] // 2) * (config.GAME_INPUT_DIM[2] // 2)

        self.dense_1 = torch.nn.Linear(self.dense_features_1, DENSE_FEATURES)
        self.relu_3 = torch.nn.LeakyReLU()
        self.dense_2 = torch.nn.Linear(DENSE_FEATURES, DENSE_FEATURES // 2)
        self.relu_4 = torch.nn.LeakyReLU()
        self.drop_2 = torch.nn.Dropout(0.5)
        self.dense_3 = torch.nn.Linear(DENSE_FEATURES // 2, 1)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.batch_1(x)
        x = self.relu_1(x)

        x = self.conv_2(x)
        x = self.batch_2(x)
        x = self.relu_2(x)

        x = self.drop_1(x)
        x = self.pool_1(x)

        x = x.view(-1, self.dense_features_1)
        x = self.dense_1(x)
        x = self.relu_3(x)
        x = self.dense_2(x)
        x = self.relu_4(x)
        x = self.dense_3(x)

        x = self.drop_2(x)

        x = self.sigmoid(x)

        return x.view(-1)

    def predict(self, x):
        pred = self.forward(x)
        return pred.item()

    def save(self):
        torch.save(self.state_dict(), self.model_path)

    def load(self):
        self.load_state_dict(torch.load(self.model_path))
        self.eval()

def get_model():
    model = GameClassifier()
    model = model.float()
    return model

def get_loss_func():
    return torch.nn.MSELoss()

def get_optimizer(model):
    return torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE,
        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    # return torch.optim.AdaDelta(
    #     model.parameters(), rho=RHO,
    #     eps=EPS, weight_decay=WEIGHT_DECAY
    # )
