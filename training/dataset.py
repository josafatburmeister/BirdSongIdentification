import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from typing import List

from data_preparation.filepaths import PathManager
from general.logging import logger


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
            raise NameError("Data files missing: ", self.data_dir)

        self.labels = pd.read_csv(self.label_file)
        self.include_noise_samples = include_noise_samples
        self.multi_label_classification = multi_label_classification
        self.class_to_idx = {}
        self.create_class_indices()

        for class_name in self.class_names():
            if class_name not in self.labels.columns:
                self.labels[class_name] = 0

        if logger:
            logger.info("\n")
            logger.info("Label distribution of %s set", split)
            for class_name in self.class_names():
                logger.info("%s : %i", class_name, self.labels[class_name].sum())
            logger.info("\n")

    def create_class_indices(self):
        categories = list(np.loadtxt(self.path_manager.categories_file(), delimiter=",", dtype=str))

        if self.include_noise_samples and not self.multi_label_classification:
            categories.append("noise")

        self.class_to_idx = {}

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
        label = self.labels.iloc[idx]

        if self.multi_label_classification:
            # create multi-hot encoding
            label_tensor = torch.zeros(self.num_classes())
            for class_name, class_id in self.class_to_idx.items():
                if label[class_name] == 1:
                    label_tensor[class_id] = 1
        else:
            for class_name, class_id in self.class_to_idx.items():
                if label[class_name] == 1:
                    label_tensor = class_id
                    # in the single label case there should be only one positive class per spectrogram, so we can stop here
                    break
            if torch.sum(label_tensor) == 0 and self.include_noise_samples:
                label_tensor[self.class_name_to_id("noise")] = 1

        if self.transform:
            image = self.transform(image)

        return image, label_tensor
