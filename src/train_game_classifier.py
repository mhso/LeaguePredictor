if __name__ == "__main__":
    import argparse
    from math import ceil
    from time import time
    import torch
    from classifier import game_classifier, game_dataset
    import config
    import matplotlib.pyplot as plt

def plot_loss(train_loss, val_loss):
    fig, (ax_1, ax_2) = plt.subplots(1, 2, sharey=True)
    #ax_1.set_ylim(bottom=0, top=0.2)
    ax_1.set_title("Training Loss")
    ax_1.plot(train_loss)
    #ax_2.set_ylim(bottom=0, top=0.2)
    ax_2.set_title("Validation Loss")
    ax_2.plot(val_loss)
    plt.show()

def train(model, dataloaders, loss_func, optimizer, epoch):
    train_loss = 0
    val_loss = 0
    try:
        print(f"=== Epoch {epoch+1} ===")
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                time_before = time()
            else:
                model.eval()
            running_loss = 0.0
            epoch_pct = 0
            for index, (x, y) in enumerate(dataloaders[phase]):
                x = x.to_dense()
                if config.TRAIN_WITH_GPU:
                    x = x.cuda()
                    y = y.cuda()
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    out = model(x)
                    if len(out[out == 1]) == len(out):
                        config.log("Gradient exploded!!", config.LOG_ERROR)
                        print(out)
                        exit(1)
                    loss = loss_func(out, y.float())
                    if phase == 'train': # update weights in training runs
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * x.size(0)
                pct = int((index / (len(dataloaders[phase].dataset) / game_classifier.BATCH_SIZE)) * 100)
                if pct > epoch_pct:
                    config.log(
                        (f"Epoch is {pct}% complete. Loss: " +
                        f"{running_loss / (index * game_classifier.BATCH_SIZE)}"),
                        end="\r")
                    epoch_pct = pct

            print()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            if phase == "val":
                config.log(f"Time taken for epoch: {time() - time_before:.2f} seconds")
                val_loss = epoch_loss
            else:
                train_loss = epoch_loss

        print()
    finally:
        config.log("=" * 50)
        config.log("Saving model to file...")
        model.cpu()
        model.save()
        if config.TRAIN_WITH_GPU:
            model.cuda()

    return train_loss, val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", type=int)
    parser.add_argument("--reload", "-r", action="store_true")
    parser.add_argument("--chunk", type=int)

    args = parser.parse_args()

    model = game_classifier.get_model()
    if args.reload:
        config.log("Loading model for further training...")
        model.load()
    loss_func = game_classifier.get_loss_func()

    if config.TRAIN_WITH_GPU:
        model = model.cuda()
        loss_func = loss_func.cuda()

    optimizer = game_classifier.get_optimizer(model)

    time_start = time()
    config.log(f"Loading and shuffling file names...")

    game_files = game_dataset.get_data_files()

    duration = time() - time_start
    config.log(f"Done in {duration:.2f} seconds.")

    chunks_total = ceil(len(game_files) / game_classifier.CHUNK_SIZE)

    train_losses = []
    val_losses = []
    chunk = 0 if args.chunk is None else args.chunk

    try:
        for epoch in range(args.epochs):
            data_split = chunk * game_classifier.CHUNK_SIZE
            files_train, files_test = game_dataset.get_data(
                data_split, data_split + game_classifier.CHUNK_SIZE,
                game_files, game_classifier.VALIDATION_SPLIT
            )

            config.log(f"Creating data loaders...")

            time_start = time()
            dataloaders = game_dataset.create_dataloaders_dict(
                game_classifier.BATCH_SIZE, files_train, files_test
            )
            duration = time() - time_start
            config.log(f"Done in {duration:.2f} seconds.")

            samples_train = len(dataloaders['train'].dataset)
            samples_val = len(dataloaders['val'].dataset)
            samples_total = samples_train + samples_val

            config.log((
                f"Chunk: From file #{data_split} to file " 
                f"#{data_split + game_classifier.CHUNK_SIZE} "
                f"({samples_total} samples)"
            ))
            config.log(f"Training Samples: {samples_train}")
            config.log(f"Validation Samples: {samples_val}")
            config.log(f"Chunk Size: {game_classifier.CHUNK_SIZE}")
            config.log(f"Batch Size: {game_classifier.BATCH_SIZE}")
            config.log(f"Learning Rate: {game_classifier.LEARNING_RATE}")

            train_loss, val_loss = train(model, dataloaders, loss_func, optimizer, epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if data_split + game_classifier.CHUNK_SIZE > len(game_files):
                chunk = 0
            else:
                chunk += 1
    finally:
        plot_loss(train_losses, val_losses)
