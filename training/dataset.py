import os
from typing import KeysView, TypeVar

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from general import FileManager, logger

T_co = TypeVar('T_co', covariant=True)


class SpectrogramDataset(Dataset):
    """
    Custom PyTorch dataset of spectrogram images.
    """

    def __init__(self, path_manager: FileManager, include_noise_samples: bool = True,
                 dataset: str = "train", multi_label_classification: bool = False,
                 undersample_noise_samples: bool = True) -> None:
        """

        Args:
            path_manager: FileManager object that manages the directory containing the spectrograms file and their
                labels.
            include_noise_samples: Whether spectrograms that are classified as "noise" during noise filtering should be
                included in the spectrogram dataset.
            dataset: Name of a dataset (e.g. train, val, or test).
            multi_label_classification: Whether a label format suitable for training multi-label models should be used.
            undersample_noise_samples: Whether the number of "noise" spectrograms should be limited to the maximum
                number of spectrograms of a sound class.
        """

        normalize = transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        transformations = [transforms.Resize(
            [224, 224]), transforms.ToTensor(), normalize]

        self.transform = transforms.Compose(transformations)

        self.path_manager = path_manager
        self.data_dir = self.path_manager.data_folder(
            dataset, "spectrograms")
        self.label_file = self.path_manager.label_file(dataset, "spectrograms")

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
            logger.info("Label distribution of %s set", dataset)
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
        """
        Creates an internal mapping of human-readable class names to class indices.

        Returns:
            None
        """

        categories = list(np.loadtxt(self.path_manager.categories_file(), delimiter=",", dtype=str))

        if self.include_noise_samples and not self.multi_label_classification:
            categories.append("noise")

        self.class_to_idx = {}

        for idx, class_name in enumerate(sorted(categories)):
            self.class_to_idx[class_name] = idx

    def class_to_id_mapping(self) -> dict:
        """

        Returns:
            A dictionary that maps human-readable class names to class indices.
        """

        return self.class_to_idx

    def id_to_class_mapping(self) -> dict:
        """

        Returns:
            A dictionary that maps class indices to human-readable class names.
        """

        return {value: key for key, value in self.class_to_idx.items()}

    def id_to_class_name(self, identifier: int):
        """
        Maps class indices to human-readable class names.

        Args:
            identifier: Index of a respective class.

        Returns:
            Name of the respective class.
        """

        return self.id_to_class_mapping()[identifier]

    def class_name_to_id(self, class_name: str):
        """
        Maps human-readable class names to class indices.

        Args:
            class_name: A class name.

        Returns:
            Index of the respective class.
        """

        return self.class_to_idx[str(class_name)]

    def class_names(self) -> KeysView:
        """

        Returns:
            Class names of the dataset.
        """

        return self.class_to_idx.keys()

    def num_classes(self) -> int:
        """

        Returns:
            Number of classes of the dataset.
        """

        return len(self.class_to_idx)

    def __len__(self) -> int:
        """

        Returns:
            Length of the dataset.
        """

        return len(self.labels)

    def __getitem__(self, idx: [int] or Tensor) -> T_co:
        """

        Args:
            idx: Index of the spectrogram to be retrieved.

        Returns:
            Spectrogram image and its class index (if "multi_label_classification" is False) or a multi-hot encoded
                label tensor (if "multi_label_classification" is True)
        """
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
