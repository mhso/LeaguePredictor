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

class DigitClassifier(torch.nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()

        self.model_path = "digit_classifier.pt"

        self.conv_1 = torch.nn.Conv2d(
            config.CHAR_INPUT_DIM[0], CONV_FILTERS, KERNEL_SIZE, stride=1, padding=1
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

        self.dense_features_1 = CONV_FILTERS * 2 * (config.CHAR_INPUT_DIM[1] // 2) * (config.CHAR_INPUT_DIM[2] // 2)

        self.dense_1 = torch.nn.Linear(self.dense_features_1, DENSE_FEATURES)
        self.relu_3 = torch.nn.LeakyReLU()

        self.pool_2 = torch.nn.MaxPool1d(2, padding=0)

        self.dense_features_2 = DENSE_FEATURES // 2

        self.dense_2 = torch.nn.Linear(self.dense_features_2, DENSE_FEATURES // 2)
        self.relu_4 = torch.nn.LeakyReLU()

        self.drop_2 = torch.nn.Dropout(0.5)

        self.dense_3 = torch.nn.Linear(self.dense_features_2, OUTPUT_DIM)
        self.softmax_1 = torch.nn.Softmax(dim=1)

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

        x = x.view(-1, 1, x.shape[1])
        x = self.pool_2(x)

        x = x.view(-1, self.dense_features_2)
        x = self.dense_2(x)
        x = self.relu_4(x)

        x = self.drop_2(x)

        x = self.dense_3(x)
        return self.softmax_1(x)

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
    model = DigitClassifier()
    model = model.float()
    return model

def get_loss_func():
    return torch.nn.CrossEntropyLoss()

def get_optimizer(model):
    return torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
