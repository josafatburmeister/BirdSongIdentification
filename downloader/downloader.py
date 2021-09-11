import os
import shutil
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from general import logger, FileManager


class Downloader:
    """
    A base class for custom dataset downloaders.
    """

    @staticmethod
    def _parse_species_list(species_list: List[str], default_sound_types: Union[List[str], Set[str]]) -> Dict[str, set]:
        """
        Parses a list of strings, where each string contains a species name and its sound types.
        The list has to be in the format ["species name, sound type name 1, sound type name 2, ...", "..."].

        Args:
            species_list: List of species and sound types in the above mentioned  format.
            default_sound_types: Default sound types that are used when no sound types are provided for a species.

        Returns:
            A dictionary that maps species names to sound types.
        """

        sound_types = set(default_sound_types)
        species = {}

        for item in species_list:
            species_name = item.split(",")[0].rstrip()

            if species_name not in species:
                species[species_name] = set()

            if len(item.split(",")) > 1:
                for sound_type in item.split(",")[1:]:
                    species[species_name].add(sound_type.lstrip().rstrip())
            else:
                species[species_name] = species[species_name].union(sound_types)

        return species

    @staticmethod
    def train_test_split(
            labels: pd.DataFrame,
            test_size: float = 0.4,
            random_state: int = 12
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Randomly splits Pandas dataframe into train and test set.

        Args:
            labels: Dataframe to be split.
            test_size: Fraction of rows that should be used for test set.
            random_state: Random state for random splitting of the dataframe.

        Returns:
            Two Pandas dataframes, the first being the train set and the second being the test set.
        """

        train_labels, test_labels = [], []
        try:
            train_labels, test_labels = train_test_split(
                labels, test_size=test_size, random_state=random_state)

        except ValueError as e:
            if "resulting train set will be empty" in str(e):
                train_labels = labels
                test_labels = []

        return train_labels, test_labels

    def __init__(self, file_manager: FileManager):
        """

        Args:
            file_manager: FileManager object that manages the output directory to be used for storing the
                downloaded datasets.
        """

        self.file_manager = file_manager

        # retrieve cached files from google cloud storage
        if self.file_manager.is_pipeline_run:
            self.file_manager.copy_cache_from_gcs("audio")
            self.file_manager.copy_cache_from_gcs("labels")

    def download_file(self, url: str, target_file: str, cache_subdir: Optional[str] = None) -> None:
        """
        Downloads file from a URL and saves it to the specified path. If the cache_subdir parameter is set, the file is
        additionally copied to a cache directory. Files that are already in the cache are not downloaded again.

        Args:
            url: URL of the file to be downloaded.
            target_file: Path where downloaded file is to be stored.
            cache_subdir: Name of a subdirectory of the cache directory where the downloaded file should be cached. If
                not existing, the subdirectory is created inside the cache directory.

        Returns:
            None
        """

        # check if file is in cache
        file_is_cached = False
        if cache_subdir:
            cached_file_path = self.file_manager.cached_file_path(cache_subdir, target_file)
            if os.path.exists(cached_file_path):
                logger.verbose("cached %s", url)
                shutil.copy(cached_file_path, target_file)
                file_is_cached = True

        # file is not in cache and needs to be downloaded
        if not file_is_cached:
            logger.verbose("download %s", url)
            response = requests.get(url, stream=True)

            if response.status_code == 200:
                response.raw.decode_content = True

                with open(target_file, "wb") as f:
                    shutil.copyfileobj(response.raw, f)

                # put file copy into cache
                if cache_subdir:
                    cached_file_path = self.file_manager.cached_file_path(cache_subdir, target_file)
                    shutil.copy(target_file, cached_file_path)
                    if self.file_manager.is_pipeline_run:
                        self.file_manager.copy_file_to_gcs_cache(cached_file_path, cache_subdir)
            else:
                raise NameError("File could not be retrieved")

    def _save_label_file(self, labels: pd.DataFrame, dataset_name: str) -> None:
        """
        Creates an audio label file for a dataset from a Pandas dataframe according to the format described in the
        Readme.

        Args:
            labels: Pandas dataframe containing the audio file labels, has to contain at least the columns "id",
                "file_path", "label", "start", and "end" (see Readme for further explanations).
            dataset_name: Name of the dataset (e.g., "train", "val" or "test).

        Returns:
            None
        """

        assert type(labels) == pd.DataFrame
        for column_name in ["id", "file_path", "label", "start", "end"]:
            assert column_name in labels.columns

        labels.to_csv(self.file_manager.label_file(dataset_name, "audio"))

    def _save_categories_file(self, categories: List[str]) -> None:
        """
        Creates categories.txt file according to the format described in the Readme.

        Args:
            categories: List of class names that may be used in the datasets (see Readme for further explanations).

        Returns:
            None
        """

        # noinspection PyTypeChecker
        np.savetxt(self.file_manager.categories_file(), np.array(categories), delimiter=",", fmt="%s")
