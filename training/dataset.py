import os
from typing import KeysView, T_co

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from general.filepaths import PathManager
from general.logging import logger


class XenoCantoSpectrograms(Dataset):
    def __init__(self, path_manager: PathManager, include_noise_samples: bool = True,
                 split: str = "train", multi_label_classification: bool = False,
                 undersample_noise_samples: bool = True) -> None:

        normalize = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        transformations = [transforms.Resize(
            [224, 224]), transforms.ToTensor(), normalize]

        self.transform = transforms.Compose(transformations)

        self.path_manager = path_manager
        self.data_dir = self.path_manager.data_folder(
            split, "spectrograms")
        self.label_file = self.path_manager.label_file(split, "spectrograms")

        if not os.path.exists(self.data_dir) or not os.path.exists(self.label_file):
            raise NameError("Data files missing: ", self.data_dir)

        self.labels = pd.read_csv(self.label_file)
        self.labels = self.labels[self.labels["file_path"].str.endswith(".png")]
        self.include_noise_samples = include_noise_samples
        self.multi_label_classification = multi_label_classification
        self.class_to_idx = {}
        self.create_class_indices()

        for class_name in self.class_names():
            if class_name not in self.labels.columns:
                self.labels[class_name] = 0

        max_samples_per_class = 0
        if logger:
            logger.info("\n")
            logger.info("Label distribution of %s set", split)
            for class_name in self.class_names():
                class_samples = self.labels[class_name].sum()
                max_samples_per_class = max(max_samples_per_class, class_samples)
                logger.info("%s : %i", class_name, class_samples)

        if undersample_noise_samples:
            noise_samples_to_include = min(int(max_samples_per_class), len(self.labels[self.labels["noise"] == 1]))
            noise_samples = self.labels[self.labels["noise"] == 1].sample(n=noise_samples_to_include, random_state=12)
            species_samples = self.labels[self.labels["noise"] != 1]
            self.labels = pd.concat([noise_samples, species_samples])

        if logger:
            number_noise_samples = len(self.labels[self.labels["noise"] == 1])
            logger.info("noise : %i", number_noise_samples)
            logger.info("Total: %i", len(self.labels))
            logger.info("\n")

    def create_class_indices(self) -> None:
        categories = list(np.loadtxt(self.path_manager.categories_file(), delimiter=",", dtype=str))

        if self.include_noise_samples and not self.multi_label_classification:
            categories.append("noise")

        self.class_to_idx = {}

        for idx, class_name in enumerate(sorted(categories)):
            self.class_to_idx[class_name] = idx

    def class_to_id_mapping(self) -> dict:
        return self.class_to_idx

    def id_to_class_mapping(self) -> dict:
        return {value: key for key, value in self.class_to_idx.items()}

    def id_to_class_name(self, identifier: int):
        return self.id_to_class_mapping()[identifier]

    def class_name_to_id(self, class_name: str):
        return self.class_to_idx[str(class_name)]

    def class_names(self) -> KeysView:
        return self.class_to_idx.keys()

    def num_classes(self) -> int:
        return len(self.class_to_idx)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: [int] or Tensor) -> T_co:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(
            self.data_dir, self.labels["file_path"].iloc[idx])

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = self.labels.iloc[idx]

        if self.multi_label_classification:
            # create multi-hot encoding
            label_tensor = torch.zeros(self.num_classes())
            for class_name, class_id in self.class_to_idx.items():
                if label[class_name] == 1:
                    label_tensor[class_id] = 1
            return image, label_tensor
        else:
            if self.include_noise_samples:
                class_label = self.class_name_to_id("noise")
            for class_name, class_id in self.class_to_idx.items():
                if label[class_name] == 1:
                    class_label = class_id
                    # in the single label case there should be only one positive class per spectrogram,
                    # so we can stop here
                    break

            return image, class_label
