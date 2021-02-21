import torch
import config

CONV_FILTERS = 32
KERNEL_SIZE = 3

LEARNING_RATE = 0.02
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

BATCH_SIZE = 64#256
VALIDATION_SPLIT = 0.7

DENSE_FEATURES = 512

OUTPUT_DIM = 10

class GameClassifier(torch.nn.Module):
    def __init__(self):
        super(GameClassifier, self).__init__()

        self.model_path = "game_classifier.pt"

        self.conv_1 = torch.nn.Conv2d(
            config.CHAR_INPUT_DIM[0], CONV_FILTERS, KERNEL_SIZE, stride=1, padding=1
        )
        self.batch_1 = torch.nn.BatchNorm2d(CONV_FILTERS)
        self.relu_1 = torch.nn.LeakyReLU()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.batch_1(x)
        x = self.relu_1(x)

        return x

    def predict(self, x):
        pred = self.forward(x)
        tensor = torch.max(pred, 1)[1]
        return [x.item() for x in tensor]

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
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
