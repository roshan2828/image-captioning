import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import CaptionDataset
from utils import build_vocab
from model import ImageCaptionModel


def caption_loss(predictions, targets):
    return F.cross_entropy(
        predictions.reshape(-1, predictions.size(-1)),
        targets.reshape(-1),
        ignore_index=0
    )


def main():
    image_folder = "images"
    captions_file = "captions.txt"
    checkpoint_path = "checkpoint.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------- BUILD VOCAB --------
    captions = []
    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            _, caption = parts
            captions.append(caption)

    vocab = build_vocab(captions)
    print("Vocabulary size:", len(vocab))

    # -------- DATASET --------
    dataset = CaptionDataset(
        image_folder=image_folder,
        captions_file=captions_file,
        vocab=vocab
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    # -------- MODEL --------
    model = ImageCaptionModel(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    start_epoch = 0
    EPOCHS = 30

    # -------- EARLY STOPPING VARS --------
    best_loss = float("inf")
    patience = 5
    counter = 0

    # -------- LOAD CHECKPOINT --------
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    # -------- TRAIN LOOP --------
    for epoch in range(start_epoch, EPOCHS):

        model.train()
        epoch_loss = 0

        for batch_idx, (images, caps) in enumerate(dataloader):

            images = images.to(device)
            caps = caps.to(device)

            optimizer.zero_grad()

            outputs = model(images, caps[:, :-1])
            loss = caption_loss(outputs, caps[:, 1:])

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch} | "
                    f"Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch} DONE | Avg Loss: {avg_loss:.4f}")

        # -------- EARLY STOPPING --------
        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0

            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved")

        else:
            counter += 1
            print(f"No improvement count: {counter}/{patience}")

        if counter >= patience:
            print("Early stopping triggered")
            break

        # -------- SAVE CHECKPOINT --------
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, checkpoint_path)

        print("Checkpoint saved")

    # -------- FINAL SAVE --------
    torch.save(model.state_dict(), "caption_model.pth")
    print("Final model saved as caption_model.pth")


if __name__ == "__main__":
    main()
