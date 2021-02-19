from sys import argv
from time import time
import torch
from classifier import digit_classifier
from classifier import digit_dataset
import config

def train(model, dataloaders, loss_func, optimizer, epochs):
    try:
        for epoch in range(epochs):
            print(f"=== Epoch {epoch+1} ===")
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                    time_before = time()
                else:
                    model.eval()
                running_loss = 0.0
                running_corrects = 0.0
                for (x, y) in dataloaders[phase]:
                    if config.TRAIN_WITH_GPU:
                        x = x.cuda()
                        y = y.cuda()
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        out = model(x)
                        loss = loss_func(out, y)
                        _, predictions = torch.max(out, 1) # gets indices of the max value, in the first dimension
                        if phase == 'train': # update weights in training runs
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * x.size(0)
                    corrects = torch.sum(predictions == y.data)
                    running_corrects += corrects
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = int((running_corrects / len(dataloaders[phase].dataset)) * 100)
                if phase == "val":
                    print(f"Time taken: {time() - time_before:.2f} seconds")

                print(f"{phase.capitalize()} Loss: {epoch_loss:.3f} - Acc: {epoch_acc}%")

        print()
    finally:
        model.save()

model = digit_classifier.get_model()
loss_func = digit_classifier.get_loss_func()

if config.TRAIN_WITH_GPU:
    model = model.cuda()
    loss_func = loss_func.cuda()

optimizer = digit_classifier.get_optimizer(model)

x_train, y_train, x_test, y_test = digit_dataset.get_data(digit_classifier.VALIDATION_SPLIT)

dataloaders = digit_dataset.create_dataloaders_dict(
    digit_classifier.BATCH_SIZE, x_train, y_train, x_test, y_test
)

if len(argv) == 1:
    print("Please provide number of epochs.")
    exit(0)

epochs = int(argv[1])

print((
    f"Training on {len(x_train)} training samples ({len(x_test)} " +
    f"validation samples) for {epochs} epochs."
))

train(model, dataloaders, loss_func, optimizer, epochs)
