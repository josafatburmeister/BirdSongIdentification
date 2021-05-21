import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class XenoCantoSpectrograms(Dataset):
    def __init__(self, path_manager, chunk_length=1000, split="train"):

        normalize = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        transformations = [transforms.Resize(
            [224, 224]), transforms.ToTensor(), normalize]

        self.transform = transforms.Compose(transformations)

        if split == "train":
            self.data_dir = path_manager.train_spectrogram_dir(chunk_length)
            self.label_file = os.path.join(
                path_manager.train_dir, "train_{}.json".format(chunk_length))

        elif split == "val":
            self.data_dir = path_manager.val_spectrogram_dir(chunk_length)
            self.label_file = os.path.join(
                path_manager.val_dir, "val_{}.json".format(chunk_length))

        elif split == "test":
            self.data_dir = path_manager.test_spectrogram_dir(chunk_length)
            self.label_file = os.path.join(
                path_manager.test_dir, "test_{}.json".format(chunk_length))
        else:
            raise NameError("Invalid split name")

        if not os.path.exists(self.data_dir) or not os.path.exists(self.label_file):
            raise NameError("Data files missing")

        self.labels = pd.read_json(self.label_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(
            self.data_dir, self.labels["file_name"].iloc[idx])

        image = Image.open(img_path).convert('RGB')
        label = self.labels.iloc[idx]

        if self.transform:
            image = self.transform(image)

        return image, label["label"]
