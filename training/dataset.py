import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from data_preparation.filepaths import PathManager


class XenoCantoSpectrograms(Dataset):
    def __init__(self, path_manager: PathManager, chunk_length: int = 1000, include_noise_samples: bool = True,
                 split: str = "train", multi_label_classification: bool = False):

        normalize = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        transformations = [transforms.Resize(
            [224, 224]), transforms.ToTensor(), normalize]

        self.transform = transforms.Compose(transformations)

        self.path_manager = path_manager
        self.data_dir = self.path_manager.data_folder(
            split, "spectrograms", chunk_length=chunk_length)
        self.label_file = self.path_manager.spectrogram_label_file(split, chunk_length=chunk_length)

        if not os.path.exists(self.data_dir) or not os.path.exists(self.label_file):
            raise NameError("Data files missing")

        self.labels = pd.read_json(self.label_file)
        self.class_to_idx = {}
        self.create_class_indices(include_noise_samples)
        self.multi_label_classification = multi_label_classification

    def create_class_indices(self, include_noise_samples):
        categories = list(np.loadtxt(self.path_manager.categories_file(), delimiter=",", dtype=str))

        if include_noise_samples:
            categories.append("noise")

        for idx, class_name in enumerate(sorted(categories)):
            self.class_to_idx[class_name] = idx

    def class_to_id_mapping(self):
        return self.class_to_idx

    def id_to_class_mapping(self):
        return {value: key for key, value in self.class_to_idx.items()}

    def id_to_class_name(self, id: int):
        return self.id_to_class_mapping()[id]

    def class_name_to_id(self, class_name: str):
        return self.class_to_idx[str(class_name)]

    def class_names(self):
        return self.class_to_idx.keys()

    def num_classes(self):
        return len(self.class_to_idx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: [int] or Tensor):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(
            self.data_dir, self.labels["file_name"].iloc[idx])

        image = Image.open(img_path).convert('RGB')
        label = self.labels.iloc[idx]["label"]
        class_id = self.class_to_idx[label]

        if self.multi_label_classification:
            # create multi-hot encoding
            label_tensor = torch.zeros(self.num_classes())
            label_tensor[class_id] = 1
        else:
            label_tensor = class_id

        if self.transform:
            image = self.transform(image)

        return image, label_tensor
