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

def train(model, dataloaders, loss_func, optimizer, epochs):
    train_loss = []
    val_loss = []
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
                epoch_pct = 0
                for index, (x, y) in enumerate(dataloaders[phase]):
                    x = x.to_dense()
                    if config.TRAIN_WITH_GPU:
                        x = x.cuda()
                        y = y.cuda()
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        out = model(x)
                        loss = loss_func(out, y.float())
                        if phase == 'train': # update weights in training runs
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * x.size(0)
                    pct = int((index / (len(dataloaders[phase].dataset) / game_classifier.BATCH_SIZE)) * 100)
                    if pct > epoch_pct:
                        print(f"{pct}% of epoch {epoch+1} complete.", end="\r", flush=True)
                        epoch_pct = pct

                print()

                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                if phase == "val":
                    config.log(f"Time taken: {time() - time_before:.2f} seconds")
                    val_loss.append(epoch_loss)
                else:
                    train_loss.append(epoch_loss)

                config.log(f"{phase.capitalize()} Loss: {epoch_loss:.3f}")

        print()
    finally:
        model.save()
        plot_loss(train_loss, val_loss)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("epochs", type=int)
    parser.add_argument("--reload", "-r", action="store_true")

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

    for chunk in range(chunks_total):
        data_split = chunk * game_classifier.CHUNK_SIZE
        config.log(f"Start training for chunk #{chunk+1}")
        files_train, files_test = game_dataset.get_data(
            data_split, data_split + game_classifier.CHUNK_SIZE,
            game_files, game_classifier.VALIDATION_SPLIT
        )

        config.log(f"Creating data loaders...")

        dataloaders = game_dataset.create_dataloaders_dict(
            game_classifier.BATCH_SIZE, files_train, files_test
        )

        config.log((
            f"Training on {len(dataloaders['train'].dataset)} training samples "
            f"({len(dataloaders['val'].dataset)} validation samples) with a batch " +
            f"size of {game_classifier.BATCH_SIZE}"
        ))

        train(model, dataloaders, loss_func, optimizer, args.epochs)
