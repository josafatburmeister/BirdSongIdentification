import datetime
import json
import os
from multiprocessing.pool import ThreadPool
from typing import List, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from downloader import Downloader, NIPS4BPlusDownloader
from general import logger, PathManager, ProgressBar


class XenoCantoDownloader(Downloader):
    xeno_canto_url = "https://www.xeno-canto.org"
    xeno_api_canto_url = "https://www.xeno-canto.org/api/2/recordings"

    # Xeno-Canto categories
    xc_quality_levels = {"A", "B", "C", "D", "E"}
    xc_sound_types = {"uncertain", "song", "subsong", "call", "alarm call", "flight call", "nocturnal flight call",
                      "begging call", "drumming", "duet", "dawn song"}
    xc_sexes = {"male", "female", "sex uncertain"}
    xc_life_stages = {"adult", "juvenile", "hatchling or nestling", "life stage uncertain"}
    xc_special_cases = {"aberrant", "mimicry/imitation", "bird in hand"}
    nips4bplus_sound_types_to_xc_sound_types = {"call": "call", "drum": "drumming", "song": "song"}

    @staticmethod
    def download_xeno_canto_page(species_name: str, page: int = 1):
        params = {"query": species_name, "page": page}

        response = requests.get(url=XenoCantoDownloader.xeno_api_canto_url, params=params)

        return response.json()

    def __init__(self, path_manager: PathManager):
        super().__init__(path_manager)

    def __metadata_cache_path(self, species_name: str):
        file_name = "{}.json".format(species_name.replace(" ", "_"))
        return os.path.join(self.path.cache("labels"), file_name)

    def __download_audio_files_by_id(self, target_dir: str, file_ids: List[str], desc: str = "Download audio files...",
                                     download_threads=25):

        progress_bar = ProgressBar(total=len(file_ids), desc=desc, position=0,
                                   is_pipeline_run=self.path.is_pipeline_run)

        url_and_filepaths = [(XenoCantoDownloader.xeno_canto_url + "/" + file_id + "/" + "download",
                              os.path.join(target_dir, file_id + ".mp3"), file_id) for file_id in file_ids]

        pool = ThreadPool(download_threads)

        def download_task(url, file_path, file_id):
            try:
                self.download_file(url, file_path, "audio")
            except Exception as e:
                progress_bar.write(
                    "Could not download file with id {}. Reason: {}".format(file_id, e))

        for _ in pool.imap_unordered(lambda x: download_task(*x), url_and_filepaths):
            progress_bar.update(1)

    # TODO fix annotations
    def __download_species_metadata(self, species_name: str) -> Tuple[Any, int]:
        metadata_file_path = self.__metadata_cache_path(species_name)

        # check if metadata file is in cache
        if os.path.exists(metadata_file_path):
            logger.verbose("Label file for %s is in cache", species_name)
            with open(metadata_file_path) as metadata_file:
                metadata = json.load(metadata_file)
                return metadata, len(metadata)
        else:
            # download first page to get total number of pages and number of recordings
            first_page = XenoCantoDownloader.download_xeno_canto_page(species_name)

            if int(first_page["numSpecies"]) != 1:
                raise NameError("Multiple species found for {}".format(species_name))

            number_of_pages = int(first_page["numPages"])
            metadata = first_page["recordings"]

            # download remaining pages
            progress_bar = ProgressBar(sequence=range(2, number_of_pages + 1),
                                       desc="Download label file for {}...".format(species_name), position=0,
                                       is_pipeline_run=self.path.is_pipeline_run)

            for page in progress_bar.iterable():
                current_page = XenoCantoDownloader.download_xeno_canto_page(
                    species_name, page)

                metadata.extend(current_page["recordings"])

            # store all labels as json file
            with open(metadata_file_path, "w") as metadata_file:
                json.dump(metadata, metadata_file, indent=2,
                          separators=(',', ':'))
            if self.path.is_pipeline_run:
                self.path.copy_file_to_gcs_cache(metadata_file_path, "labels")

            return metadata, first_page["numRecordings"]

    def create_datasets(self,
                        species_list: Optional[List[str]] = None,
                        use_nips4b_species_list: bool = True,
                        maximum_samples_per_class: int = 100,
                        test_size: float = 0.4,
                        min_quality: str = "E",
                        sound_types: Optional[List[str]] = None,
                        sexes: Optional[List[str]] = None,
                        life_stages: Optional[List[str]] = None,
                        exclude_special_cases: bool = True,
                        maximum_number_of_background_species: Optional[int] = None,
                        maximum_recording_length: int = None,
                        clear_audio_cache: bool = False,
                        clear_label_cache: bool = False,
                        random_state: int = 12
                        ) -> None:
        if use_nips4b_species_list or not species_list:
            nips4bplus_downloader = NIPS4BPlusDownloader(self.path)
            species_list = nips4bplus_downloader.download_nips4b_species_list()
            # FIXME multiline lambda ist problematisch
            species_list["species_sound_type"] = species_list.apply(
                lambda sl_row: sl_row["Scientific_name"] + ", " + sl_row["sound_type"] if sl_row["sound_type"] else
                sl_row[
                    "Scientific_name"], axis=1)
            species_list = species_list["species_sound_type"].tolist()
            species_list = [item for item in species_list if item]
        if len(species_list) < 1:
            raise ValueError("Empty species list")
        if maximum_samples_per_class < 3:
            raise ValueError(
                "At least three samples are needed for each class")
        if not life_stages:
            life_stages = ["adult", "juvenile",
                           "hatchling or nestling", "life stage uncertain"]
        if not sexes:
            sexes = ["male", "female", "sex uncertain"]
        if not sound_types:
            sound_types = ["song", "call"]

        if min_quality not in XenoCantoDownloader.xc_quality_levels:
            raise ValueError("Invalid quality level for Xeno-Canto database")
        if not set(sound_types).issubset(XenoCantoDownloader.xc_sound_types):
            raise ValueError("Invalid sound type for Xeno-Canto database")
        if not set(sexes).issubset(XenoCantoDownloader.xc_sexes):
            raise ValueError("Invalid sex for Xeno-Canto database")
        if not set(life_stages).issubset(XenoCantoDownloader.xc_life_stages):
            raise ValueError("Invalid life stage for Xeno-Canto database")

        if clear_audio_cache:
            self.path.clear_cache("audio")
        if clear_label_cache:
            self.path.clear_cache("labels")

        train_frames = []
        test_frames = []
        val_frames = []
        categories = []

        species_sounds_dict = XenoCantoDownloader.parse_species_list(species_list, XenoCantoDownloader.xc_sound_types)
        for species_name, species_sound_types in species_sounds_dict.items():
            try:
                labels, _ = self.__download_species_metadata(species_name)
            except Exception as e:
                logger.info(f"{e}")
                logger.info("Skipping class %s", species_name)
                continue

            labels = pd.DataFrame.from_dict(labels)

            # filter samples by quality
            if min_quality < "E":
                labels = labels[labels["q"] <= min_quality]

            selected_sound_types = species_sound_types.intersection(set(sound_types))

            for sound_type in selected_sound_types:
                categories.append(f"{species_name.replace(' ', '_')}_{sound_type}")

            logger.verbose("Sound types for %s: %s", species_name, str(selected_sound_types))

            # filter samples by soundtype since some Xeno-Canto files miss some type annotations,
            # only filter if a true subset of the categories was selected
            if selected_sound_types < XenoCantoDownloader.xc_sound_types:
                type_search_string = "|".join(selected_sound_types)
                type_exclude_string = "|".join(
                    XenoCantoDownloader.xc_sound_types - set(selected_sound_types))

                labels = labels[(labels["type"].str.contains(type_search_string))
                                & (~labels["type"].str.contains(type_exclude_string))]

            # filter samples by sex since some Xeno-Canto files miss some type annotations,
            # only filter if a true subset of the categories was selected
            if set(sexes) < XenoCantoDownloader.xc_sexes:
                sex_search_string = "|".join(sexes)
                labels = labels[labels["type"].str.contains(sex_search_string)]

            # filter samples by life stage since some Xeno-Canto files miss some type annotations,
            # only filter if a true subset of the categories was selected
            if set(life_stages) < XenoCantoDownloader.xc_life_stages:
                life_stage_search_string = "|".join(life_stages)
                labels = labels[labels["type"].str.contains(
                    life_stage_search_string)]

            # remove special cases
            if exclude_special_cases:
                special_cases_stage_search_string = "|".join(
                    XenoCantoDownloader.xc_special_cases)
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
            labels["file_name"] = labels["id"] + ".mp3"

            labels["sound_type"] = ""
            for idx, row in labels.iterrows():
                for sound_type in sound_types:
                    if sound_type in row["type"]:
                        labels.loc[idx, "sound_type"] = sound_type
                        break
            labels["label"] = labels["gen"] + "_" + labels["sp"] + "_" + labels["sound_type"]

            def st(length, time_format):
                return datetime.datetime.strptime(length, time_format)

            labels["duration"] = labels["length"].apply(
                lambda length: st(length, "%M:%S") if length.count(':') == 1 else st(length, "%H:%M:%S")
            )

            labels["start"] = 0
            labels["end"] = labels["duration"].apply(
                lambda duration: duration.hour * 60 * 60 * 1000 + duration.minute * 60 * 1000 + duration.second * 1000
            )
            if maximum_recording_length:
                labels = labels[labels["end"] < maximum_recording_length * 1000]

            # select relevant columns
            labels = labels[["id", "file_name", "start", "end", "label", "sound_type"]]
            labels = labels[labels["end"] > 0]

            for sound_type in selected_sound_types:
                # obtain random subset if maximum_samples_per_class is set
                if len(labels[labels["sound_type"] == sound_type]) > maximum_samples_per_class:
                    label_subset, _ = train_test_split(
                        labels[labels["sound_type"] == sound_type], train_size=maximum_samples_per_class,
                        random_state=random_state)
                else:
                    label_subset = labels[labels["sound_type"] == sound_type]

                # create train, test and val splits
                train_labels, test_labels = self.train_test_split(
                    label_subset, test_size=test_size, random_state=random_state)

                val_labels, test_labels = self.train_test_split(
                    test_labels, test_size=0.5, random_state=random_state)

                logger.info("%d train samples for %s_%s", len(train_labels), species_name, sound_type)
                logger.info("%d val samples for %s_%s", len(val_labels), species_name, sound_type)
                logger.info("%d test samples for %s_%s", len(test_labels), species_name, sound_type)

                if len(train_labels) == 0:
                    logger.info("No training samples for class %s", species_name)
                else:
                    train_frames.append(train_labels)

                if len(val_labels) == 0:
                    logger.info("No validation samples for class %s", species_name)
                else:
                    val_frames.append(val_labels)
                if len(test_labels) == 0:
                    logger.info("No test samples for class %s", species_name)
                else:
                    test_frames.append(test_labels)

        # noinspection PyTypeChecker
        np.savetxt(self.path.categories_file(), np.array(categories), delimiter=",", fmt="%s")

        # save label files
        for split_name, frames in [("train", train_frames), ("val", val_frames), ("test", test_frames)]:
            if len(frames) == 0:
                raise NameError(f"Empty {split_name} set")
            labels = pd.concat(frames)

            self.save_label_file(labels, split_name)

            # clear data folders
            PathManager.empty_dir(self.path.data_folder(split_name, "audio"))

            # download audio files
            self.__download_audio_files_by_id(
                self.path.data_folder(split_name, "audio"), labels["id"], f"Download {split_name} set..."
            )
