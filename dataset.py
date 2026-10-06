import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CaptionDataset(Dataset):
    def __init__(self, image_folder, captions_file, vocab, max_len=20):
        self.image_folder = image_folder
        self.vocab = vocab
        self.max_len = max_len

        self.data = []

        # all image filenames present
        available_images = set(os.listdir(image_folder))

        with open(captions_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # SPLIT ON FIRST WHITESPACE (THIS IS THE KEY FIX)
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue

                img, caption = parts

                # clean image name
                img = img.strip().split('#')[0]
                caption = caption.strip()

                # ensure image exists
                if img not in available_images:
                    continue

                self.data.append((img, caption))

        print("Total valid samples loaded:", len(self.data))

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, caption = self.data[idx]

        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        tokens = caption.lower().split()
        tokens = ["<start>"] + tokens + ["<end>"]

        caption_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
        caption_ids = caption_ids[:self.max_len]

        while len(caption_ids) < self.max_len:
            caption_ids.append(self.vocab["<pad>"])

        return image, torch.tensor(caption_ids)
