import os
import pandas as pd
import requests
import shutil
from sklearn.model_selection import train_test_split
from typing import Optional, Tuple

from general.filepaths import PathManager
from general.logging import logger

class Downloader:
    """
    A base class for custom dataset downloaders.
    """

    @staticmethod
    def train_test_split(labels: pd.DataFrame, test_size: float = 0.4, random_state: int = 12) -> Tuple[
        pd.DataFrame, pd.DataFrame]:
        """
        Splits pandas dataframe into train and test set.

        param: labels: Labels to split
        param: test_size: Fraction of labels that should be used for test set
        param: random_state: Random state for data splitting
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

    def __init__(self, path_manager: PathManager):
        """
        param: path_manager: PathManager instance that manages the target folder
        """

        self.path = path_manager

        # retrieve cached files from google cloud storage
        if self.path.is_pipeline_run:
            self.path.copy_cache_from_gcs("audio")
            self.path.copy_cache_from_gcs("labels")

    def __del__(self):
        # clean up
        if self.path.is_pipeline_run:
            PathManager.empty_dir(self.path.cache_dir)

    def download_file(self, url: str, target_file: str, cache_subdir: Optional[str] = None):
        """
        Downloads file from given URL and saves it in target path. If cache_dir parameter is set the downloaded file is cached to speedup later download request of the same file.

        param: url: url of file to download
        param: target_file: path where downloaded file should be stored
        param: cache_subdir: Name of a subfolder of the cache folder where the downloaded file should be cached. If not existing, the subfolder is created inside the cache folder.
        """

        if cache_subdir:
            cached_file_path = self.path.cached_file_path(cache_subdir, target_file)
        # check if file is in cache
        if cache_subdir and os.path.exists(cached_file_path):
            logger.verbose("cached %s", url)
            shutil.copy(cached_file_path, target_file)
        # download file
        else:
            logger.verbose("download %s", url)
            response = requests.get(url, stream=True)

            if response.status_code == 200:
                response.raw.decode_content = True

                with open(target_file, "wb") as f:
                    shutil.copyfileobj(response.raw, f)

                # put file copy into cache
                if cache_subdir:
                    shutil.copy(target_file, cached_file_path)
                    if self.path.is_pipeline_run:
                        self.path.copy_file_to_gcs_cache(cached_file_path, cache_subdir)
            else:
                raise NameError("File couldn\'t be retrieved")