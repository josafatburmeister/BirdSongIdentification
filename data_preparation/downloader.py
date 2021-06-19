import importlib.resources as pkg_resources
import json
import os
import pandas as pd
import requests
import shutil
from sklearn.model_selection import train_test_split
from typing import List, Optional

from data_preparation.filepaths import PathManager
from general.logging import logger, ProgressBar

import data_preparation

class XenoCantoDownloader:
    def __init__(self, path_manager: PathManager):
        self.xeno_canto_url = "https://www.xeno-canto.org"
        self.xeno_api_canto_url = "https://www.xeno-canto.org/api/2/recordings"

        # Xeno-Canto categories
        self.xc_quality_levels = {"A", "B", "C", "D", "E"}
        self.xc_sound_types = {"uncertain", "song", "subsong", "call", "alarm call", "flight call",
                               "nocturnal flight call", "begging call", "drumming", "duet", "dawn song"}
        self.xc_sexes = {"male", "female", "sex uncertain"}
        self.xc_life_stages = {"adult", "juvenile",
                               "hatchling or nestling", "life stage uncertain"}
        self.xc_special_cases = {"aberrant",
                                 "mimicry/imitation", "bird in hand"}

        self.path = path_manager

        # retrieve cached files from google cloud storage
        if self.path.is_pipeline_run:
            self.path.copy_cache_from_gcs("audio")
            self.path.copy_cache_from_gcs("labels")

    def metadata_cache_path(self, species_name: str):
        file_name = "{}.json".format(species_name.replace(" ", "_"))
        return os.path.join(self.path.cache("labels"), file_name)

    def download_file(self, url: str, target_file: str, cache_subdir: Optional[str] = None):
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

    def download_audio_file(self, url: str, target_file: str):
        self.download_file(url, target_file, "audio")

    def download_audio_files_by_id(self, target_dir: str, file_ids: List[str], desc: str = "Download audio files..."):
        progress_bar = ProgressBar(
            file_ids, desc=desc, position=0, is_pipeline_run=self.path.is_pipeline_run)

        for file_id in progress_bar.iterable():
            url = self.xeno_canto_url + "/" + file_id + "/" + "download"
            file_path = os.path.join(target_dir, file_id + ".mp3")
            try:
                self.download_audio_file(url, file_path)
            except Exception:
                 progress_bar.write(
                     "Could not download file with id {}".format(file_id))

    def download_xeno_canto_page(self, species_name: str, page: int = 1):
        params = {"query": species_name, "page": page}

        response = requests.get(url=self.xeno_api_canto_url, params=params)

        return response.json()

    def download_species_metadata(self, species_name: str):
        metadata_file_path = self.metadata_cache_path(species_name)

        # check if metadata file is in cache
        if os.path.exists(metadata_file_path):
            logger.verbose("Label file for %s is in cache", species_name)
            with open(metadata_file_path) as metadata_file:
                metadata = json.load(metadata_file)
                return metadata, len(metadata)
        else:
            # download first page to get total number of pages and number of recordings
            first_page = self.download_xeno_canto_page(species_name)

            if int(first_page["numSpecies"]) != 1:
                raise NameError(
                    "Multiple species found for {}".format(species_name))

            number_of_pages = int(first_page["numPages"])
            metadata = first_page["recordings"]

            # download remaining pages
            progress_bar = ProgressBar(range(2, number_of_pages + 1),
                                       desc="Download label file for {}...".format(species_name), position=0,
                                       is_pipeline_run=self.path.is_pipeline_run)

            for page in progress_bar.iterable():
                current_page = self.download_xeno_canto_page(
                    species_name, page)

                metadata.extend(current_page["recordings"])

            # store all labels as json file
            with open(metadata_file_path, "w") as metadata_file:
                json.dump(metadata, metadata_file, indent=2,
                          separators=(',', ':'))
                if self.path.is_pipeline_run:
                    self.path.copy_file_to_gcs_cache(metadata_file_path, "labels")

            return metadata, first_page["numRecordings"]

    def create_datasets(self, species_list: Optional[List[str]] = None, use_nips4b_species_list: bool = True,
                        maximum_samples_per_class: int = 100, test_size: float = 0.35,
                        min_quality: str = "E", sound_types: Optional[List[str]] = None, sexes: Optional[List[str]] = None,
                        life_stages: Optional[List[str]] = None, exclude_special_cases: bool = True,
                        maximum_number_of_background_species: Optional[int] = None,
                        clear_audio_cache: bool = False, clear_label_cache: bool = False, random_state: int = 12):
        if use_nips4b_species_list or species_list == None:
            with pkg_resources.path(data_preparation, 'nips4b_species_list.csv') as species_file:
                species_list = self.load_species_list_from_file(str(species_file))
        if len(species_list) < 1:
            raise ValueError("Empty species list")
        if maximum_samples_per_class < 3:
            raise ValueError(
                "At least three samples are needed for each class")
        if life_stages is None:
            life_stages = ["adult", "juvenile",
                           "hatchling or nestling", "life stage uncertain"]
        if sexes is None:
            sexes = ["male", "female", "sex uncertain"]
        if sound_types is None:
            sound_types = ["song"]

        if min_quality not in self.xc_quality_levels:
            raise ValueError("Invalid quality level for Xeno-Canto database")
        if not set(sound_types).issubset(self.xc_sound_types):
            raise ValueError("Invalid sound type for Xeno-Canto database")
        if not set(sexes).issubset(self.xc_sexes):
            raise ValueError("Invalid sex for Xeno-Canto database")
        if not set(life_stages).issubset(self.xc_life_stages):
            raise ValueError("Invalid life stage for Xeno-Canto database")

        if clear_audio_cache:
            self.path.clear_cache("audio")
        if clear_label_cache:
            self.path.clear_cache("labels")

        train_frames = []
        test_frames = []
        val_frames = []

        for species_name in species_list:
            try:
                labels, _ = self.download_species_metadata(species_name)
            except Exception as e:
                print(e)
                print("Skipping class", species_name)
                continue

            labels = pd.DataFrame.from_dict(labels)

            # filter samples by quality
            if min_quality < "E":
                labels = labels[labels["q"] <= min_quality]

            # filter samples by soundtype
            # since some Xeno-Canto files miss some type annotations, only filter if a true subset of the categories was selected
            if set(sound_types) < self.xc_sound_types:
                type_search_string = "|".join(sound_types)
                type_exclude_string = "|".join(
                    self.xc_sound_types - set(sound_types))

                labels = labels[(labels["type"].str.contains(
                    type_search_string)) & (~labels["type"].str.contains(
                    type_exclude_string))]

            # filter samples by sex
            # since some Xeno-Canto files miss some type annotations, only filter if a true subset of the categories was selected
            if set(sexes) < self.xc_sexes:
                sex_search_string = "|".join(sexes)
                labels = labels[labels["type"].str.contains(sex_search_string)]

            # filter samples by life stage
            # since some Xeno-Canto files miss some type annotations, only filter if a true subset of the categories was selected
            if set(life_stages) < self.xc_life_stages:
                life_stage_search_string = "|".join(life_stages)
                labels = labels[labels["type"].str.contains(
                    life_stage_search_string)]

            # remove special cases
            if exclude_special_cases:
                special_cases_stage_search_string = "|".join(
                    self.xc_special_cases)
                labels = labels[~labels["type"].str.contains(
                    special_cases_stage_search_string)]

            labels["background_species"] = labels["also"].apply(
                lambda x: len(x))

            # remove samples with background species
            if maximum_number_of_background_species:
                labels = labels[labels["background_species"]
                                <= maximum_number_of_background_species]

            if len(labels) == 0:
                raise NameError(
                    "There are no training samples for class {}".format(species_name))

            # create class labels
            labels["sound_type"] = ""
            for idx, row in labels.iterrows():
                for sound_type in sound_types:
                    if sound_type in row["type"]:
                        labels.loc[idx, "sound_type"] = sound_type
                        break
            labels["label"] = labels["gen"] + "_" + \
                              labels["sp"] + "_" + labels["sound_type"]

            # select relevant columns
            labels = labels[["id", "label", "q",
                             "sound_type", "background_species"]]

            # obtain random subset if maximum_samples_per_class is set
            if len(labels) > maximum_samples_per_class:
                labels, _ = train_test_split(
                    labels, train_size=maximum_samples_per_class, random_state=random_state)

            # create train, test and val splits
            train_labels, test_labels = self.train_test_split(
                labels, test_size=test_size, random_state=random_state)

            val_labels, test_labels = self.train_test_split(
                test_labels, test_size=test_size, random_state=random_state)

            if len(train_labels) == 0:
                print("No training samples for class", species_name)
            else:
                train_frames.append(train_labels)

            if len(val_labels) == 0:
                print("No validation samples for class", species_name)
            else:
                val_frames.append(val_labels)
            if len(test_labels) == 0:
                print("No test samples for class", species_name)
            else:
                test_frames.append(test_labels)

        # save label files
        if len(train_frames) == 0:
            raise NameError("Empty training set")
        if len(val_frames) == 0:
            raise NameError("Empty validation set")
        if len(test_frames) == 0:
            raise NameError("Empty test set")

        training_set = pd.concat(train_frames)
        validation_set = pd.concat(val_frames)
        test_set = pd.concat(test_frames)
        training_set.to_json(self.path.audio_label_file(
            "train"), "records", indent=4)
        validation_set.to_json(
            self.path.audio_label_file("val"), "records", indent=4)
        test_set.to_json(self.path.audio_label_file(
            "test"), "records", indent=4)

        # clear data folders
        PathManager.empty_dir(self.path.data_folder("train", "audio"))
        PathManager.empty_dir(self.path.data_folder("val", "audio"))
        PathManager.empty_dir(self.path.data_folder("test", "audio"))

        # download audio files
        self.download_audio_files_by_id(
            self.path.data_folder("train", "audio"), training_set["id"], "Download training set...")

        self.download_audio_files_by_id(
            self.path.data_folder("val", "audio"), validation_set["id"], "Download validation set...")

        self.download_audio_files_by_id(
            self.path.data_folder("test", "audio"), test_set["id"], "Download test set...")

    def load_species_list_from_file(self, file_path: str, column_name: str = "Scientific_name"):
        if not file_path.endswith(".csv") and not file_path.endswith(".json"):
            return []

        if ".csv" in file_path:
            species = pd.read_csv(file_path)

        elif ".json" in file_path:
            species = pd.read_json(file_path)

        return list(species[column_name])

    def train_test_split(self, labels, test_size: float = 0.35, random_state: int = 12):
        try:
            train_labels, test_labels = train_test_split(
                labels, test_size=test_size, random_state=random_state)

        except ValueError as e:
            if "resulting train set will be empty" in str(e):
                train_labels = labels
                test_labels = []

        return train_labels, test_labels
