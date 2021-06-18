import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from data_preparation.filepaths import PathManager


class XenoCantoSpectrograms(Dataset):
    def __init__(self, path_manager: PathManager, chunk_length: int = 1000, split: str = "train"):

        normalize = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        transformations = [transforms.Resize(
            [224, 224]), transforms.ToTensor(), normalize]

        self.transform = transforms.Compose(transformations)

        self.data_dir = path_manager.data_folder(
            split, "spectrograms", chunk_length=chunk_length)
        self.label_file = path_manager.spectrogram_label_file(split,
                                                              chunk_length=chunk_length)

        if not os.path.exists(self.data_dir) or not os.path.exists(self.label_file):
            raise NameError("Data files missing")

        self.labels = pd.read_json(self.label_file)
        self.create_class_indices()

    def create_class_indices(self):
        self.class_to_idx = {}

        class_names = self.labels.label.unique()

        for idx, class_name in enumerate(class_names):
            self.class_to_idx[class_name] = idx

    def class_to_id_mapping(self):
        return self.class_to_idx

    def id_to_class_mapping(self):
        return {value: key for key, value in self.class_to_idx.items()}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(
            self.data_dir, self.labels["file_name"].iloc[idx])

        image = Image.open(img_path).convert('RGB')
        label = self.labels.iloc[idx]["label"]
        class_id = self.class_to_idx[label]

        label_tensor = class_id

        if self.transform:
            image = self.transform(image)

        return image, label_tensor
