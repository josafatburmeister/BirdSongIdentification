import datetime
import json
import os
from multiprocessing.pool import ThreadPool
from typing import List, Optional, Tuple

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from downloader import Downloader, NIPS4BPlusDownloader
from general import logger, FileManager, ProgressBar
from general.custom_types import JSON


class XenoCantoDownloader(Downloader):
    """
    Downloads datasets from Xeno-Canto.

    See  Willem-Pier Vellinga and Robert Planque. “The Xeno-canto collection and its relation to sound recognition
    and classification”. In: Working Notes of CLEF 2015 - Conference and Labs of the Evaluation Forum (Toulouse,
    France). Ed. by Linda Cappellato et al. Vol. 1391. CEUR Workshop Proceedings. CEUR, Sept. 2015, pp. 1–10. URL:
    http://ceur-ws.org/Vol-1391/166-CR.pdf.
    """

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
    def __download_xeno_canto_page(species_name: str, page: int = 1) -> JSON:
        """
        The Xeno-Canto API allows to query audio metadata using a keyword search. The search results are returned
        paginated. This method searches for a specific bird species and downloads the specified page of resulting
        metadata.

        Args:
            species_name: Scientific name of the bird species for which the audio metadata is to be searched.
            page: Number of the result page to download.

        Returns:
            Xeno-Canto metadata page in JSON format.
        """

        params = {"query": species_name, "page": page}

        response = requests.get(url=XenoCantoDownloader.xeno_api_canto_url, params=params)

        return response.json()

    def __init__(self, file_manager: FileManager):
        """

        Args:
            file_manager: FileManager object that manages the output directory to be used for storing the downloaded
                datasets.
        """

        super().__init__(file_manager)

    def __metadata_cache_path(self, species_name: str) -> str:
        """
        Computes the path where the metadata file for a given bird species should be cached.

        Args:
            species_name: Scientific name of the bird species.

        Returns:
            Path of the metadata file in cache.
        """

        file_name = "{}.json".format(species_name.replace(" ", "_"))
        return os.path.join(self.file_manager.cache("labels"), file_name)

    def __download_audio_files_by_id(self, target_dir: str, file_ids: List[str], desc: str = "Download audio files...",
                                     download_threads=25) -> None:
        """
        Takes a list of Xeno-Canto file IDs and downloads the corresponding files from Xeno-Canto.

        Args:
            target_dir: Path of the directory where the downloaded files should be stored.
            file_ids: List of Xeno-Canto file IDs of the files to be downloaded.
            desc: Description of the download task.
            download_threads: Number of threads to be used for parallelization of data download.

        Returns:
            None
        """

        progress_bar = ProgressBar(total=len(file_ids), desc=desc, position=0,
                                   is_pipeline_run=self.file_manager.is_pipeline_run)

        url_and_filepaths = [(XenoCantoDownloader.xeno_canto_url + "/" + file_id + "/" + "download",
                              os.path.join(target_dir, file_id + ".mp3"), file_id) for file_id in file_ids]

        pool = ThreadPool(download_threads)

        def download_task(url, file_path, file_id):
            try:
                self.download_file(url, file_path, "audio")
            except Exception as e:
                progress_bar.write(f"Could not download file with id {file_id}. Reason: {e}.")

        for _ in pool.imap_unordered(lambda x: download_task(*x), url_and_filepaths):
            progress_bar.update(1)

    def __download_species_metadata(self, species_name: str) -> Tuple[JSON, int]:
        """
        Downloads metadata of all audio recordings available for a given bird species in Xeno-Canto.

        Args:
            species_name: Scientific name of the bird species for which the audio metadata is to be downloaded.

        Returns:
            Xeno-Canto metadata in JSON format and number of recordings available for the bird species.
        """

        metadata_file_path = self.__metadata_cache_path(species_name)

        # check if metadata file is in cache
        if os.path.exists(metadata_file_path):
            logger.verbose("Label file for %s is in cache", species_name)
            with open(metadata_file_path) as metadata_file:
                metadata = json.load(metadata_file)
                return metadata, len(metadata)
        else:
            # download first page to get total number of pages and number of recordings
            first_page = XenoCantoDownloader.__download_xeno_canto_page(species_name)

            if int(first_page["numSpecies"]) != 1:
                raise NameError("Multiple species found for {}".format(species_name))

            number_of_pages = int(first_page["numPages"])
            metadata = first_page["recordings"]

            # download remaining pages
            progress_bar = ProgressBar(sequence=range(2, number_of_pages + 1),
                                       desc="Download label file for {}...".format(species_name), position=0,
                                       is_pipeline_run=self.file_manager.is_pipeline_run)

            for page in progress_bar.iterable():
                current_page = XenoCantoDownloader.__download_xeno_canto_page(species_name, page)

                metadata.extend(current_page["recordings"])

            # store all labels as json file
            with open(metadata_file_path, "w") as metadata_file:
                json.dump(metadata, metadata_file, indent=2,
                          separators=(',', ':'))
            if self.file_manager.is_pipeline_run:
                self.file_manager.copy_file_to_gcs_cache(metadata_file_path, "labels")

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
        """
        Creates training, validation and test sets from Xeno-Canto recordings for a given list of bird species and sound
        types.
        The species_list has to be in the format ["species name, sound type name 1, sound type name 2, ...", "..."].

        Args:
            species_list: List of species and sound types in the above mentioned  format.
            use_nips4b_species_list: Whether the species list of the NIPS4B dataset should be used (if set to true, the
                provided species list can be empty)
            maximum_samples_per_class: Maximum number of recordings per class.
            test_size: Percentage of recordings that should be used for model testing and validation (validation and
                test set get one half of the samples each).
            min_quality: Minimum quality of the audio recordings to be included in the datasets.
            sound_types: List of sound types to include in the datasets.
            sexes: List of bird sexes to include in the datasets.
            life_stages: List of bird life stages to include in the datasets.
            exclude_special_cases: Whether special cases (e.g. birds imitating other birds) should be excluded from the
                datasets.
            maximum_number_of_background_species: Maximum number of background species of the audio recordings to be
                included in the datasets.
            maximum_recording_length: Maximum length of the recordings to be included in the datasets.
            clear_audio_cache: Whether the audio cache should be cleared before downloading the datasets.
            clear_label_cache: Whether the label cache should be cleared before downloading the datasets.
            random_state: Random State for random partitioning of the recordings into the training, validation and test
                sets.

        Returns:
            None
        """

        if use_nips4b_species_list or not species_list:
            nips4bplus_downloader = NIPS4BPlusDownloader(self.file_manager)
            species_list = nips4bplus_downloader.download_nips4b_species_list()

            def get_species_sound_type(row):
                if row["sound_type"]:
                    return row["Scientific_name"] + ", " + row["sound_type"]
                else:
                    return row["Scientific_name"]

            species_list["species_sound_type"] = species_list.apply(get_species_sound_type, axis=1)
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
            self.file_manager.clear_cache("audio")
        if clear_label_cache:
            self.file_manager.clear_cache("labels")

        train_frames = []
        test_frames = []
        val_frames = []
        categories = []

        species_sounds_dict = XenoCantoDownloader._parse_species_list(species_list, XenoCantoDownloader.xc_sound_types)
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
            labels["file_path"] = labels["id"] + ".mp3"

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
            labels = labels[["id", "file_path", "start", "end", "label", "sound_type"]]
            labels = labels[labels["end"] > 0]

            for sound_type in selected_sound_types:
                # obtain random subset if maximum_samples_per_class is set
                if len(labels[labels["sound_type"] == sound_type]) > maximum_samples_per_class:
                    label_subset, _ = train_test_split(
                        labels[labels["sound_type"] == sound_type], train_size=maximum_samples_per_class,
                        random_state=random_state)
                else:
                    label_subset = labels[labels["sound_type"] == sound_type]

                # create train, test and val sets
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

        self._save_categories_file(categories)

        # save label files
        for dataset_name, frames in [("train", train_frames), ("val", val_frames), ("test", test_frames)]:
            if len(frames) == 0:
                raise NameError(f"Empty {dataset_name} set")
            labels = pd.concat(frames)

            self._save_label_file(labels, dataset_name)

            # clear data folders
            FileManager.empty_dir(self.file_manager.data_folder(dataset_name, "audio"))

            # download audio files
            self.__download_audio_files_by_id(
                self.file_manager.data_folder(dataset_name, "audio"), labels["id"], f"Download {dataset_name} set..."
            )
