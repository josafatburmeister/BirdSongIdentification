import json
import os
import pandas as pd
import requests
import shutil
from sklearn.model_selection import train_test_split
import tqdm


class XenoCantoDownloader():
    def __init__(self, target_dir):
        self.xeno_canto_url = "https://www.xeno-canto.org"
        self.xeno_api_canto_url = "https://www.xeno-canto.org/api/2/recordings"

        # Xeno-Canto categories
        self.xc_quality_levels = set(["A", "B", "C", "D", "E"])
        self.xc_sound_types = set(["uncertain", "song", "subsong", "call", "alarm call", "flight call",
                                  "nocturnal flight call", "begging call", "drumming", "duet", "dawn song"])
        self.xc_sexes = set(["male", "female", "sex uncertain"])
        self.xc_life_stages = set(["adult", "juvenile",
                                   "hatchling or nestling", "life stage uncertain"])
        self.xc_special_cases = set(["aberrant",
                                     "mimicry/imitation", "bird in hand"])

        self.cache_dir = os.path.join(target_dir, "cache")
        self.audio_cache_dir = os.path.join(self.cache_dir, "audio")
        self.label_cache_dir = os.path.join(self.cache_dir, "labels")

        self.train_dir = os.path.join(target_dir, "train")
        self.train_audio_dir = os.path.join(self.train_dir, "audio")
        self.test_dir = os.path.join(target_dir, "test")
        self.test_audio_dir = os.path.join(self.test_dir, "audio")
        self.val_dir = os.path.join(target_dir, "val")
        self.val_audio_dir = os.path.join(self.val_dir, "audio")

        # create outer directories
        self.ensure_dirs([self.cache_dir, self.train_dir,
                         self.test_dir, self.val_dir])

        # create nested directories
        self.ensure_dirs([self.audio_cache_dir, self.label_cache_dir, self.train_audio_dir, self.test_audio_dir,
                          self.val_audio_dir])

    def ensure_dir(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def ensure_dirs(self, dir_paths):
        for dir_path in dir_paths:
            self.ensure_dir(dir_path)

    def metadata_cache_path(self, species_name):
        file_name = "{}.json".format(species_name.replace(" ", "_"))
        return os.path.join(self.label_cache_dir, file_name)

    def download_file(self, url, target_file, cache_dir=None):
        file_name = os.path.basename(target_file)

        cached_file_path = os.path.join(self.audio_cache_dir, file_name)

        # check if file is in cache
        if cache_dir and os.path.exists(cached_file_path):
            shutil.copy(cached_file_path, target_file)
        # download file
        else:
            response = requests.get(url, stream=True)

            if response.status_code == 200:
                response.raw.decode_content = True

                with open(target_file, "wb") as f:
                    shutil.copyfileobj(r.raw, f)

                # put file copy into cache
                if cache_dir:
                    shutil.copy(target_file, cached_file_path)
            else:
                raise NameError("File couldn\'t be retrieved")

    def download_audio_file(self, url, target_file):
        self.download_file(url, target_file, self.audio_cache_dir)

    def download_audio_files_by_id(self, target_dir, file_ids, desc="Download audio files..."):
        progress_bar = tqdm.tqdm(
            total=len(file_ids), desc=desc, position=0)

        for file_id in file_ids:
            url = self.xeno_canto_url + "/" + file_id + "/" + "download"
            file_path = os.path.join(target_dir, file_id + ".mp3")
            try:
                self.download_audio_file(url, file_path)
            except Exception:
                progress_bar.write(
                    "Could not download file with id {}".format(file_id))

            progress_bar.update(1)

    def download_xeno_canto_page(self, species_name, page=1):
        params = {"query": species_name, "page": page}

        response = requests.get(url=self.xeno_api_canto_url, params=params)

        return response.json()

    def download_species_metadata(self, species_name):
        metadata_file_path = self.metadata_cache_path(species_name)

        # check if metadata file is in cache
        if os.path.exists(metadata_file_path):
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
            progress_bar = tqdm.tqdm(
                total=number_of_pages, desc="Download label file for {}...".format(species_name), position=0)
            progress_bar.update(1)

            for page in range(2, number_of_pages+1):
                current_page = self.download_xeno_canto_page(
                    species_name, page)

                metadata.extend(current_page["recordings"])

                progress_bar.update(1)

            # store all labels as json file
            with open(metadata_file_path, "w") as metadata_file:
                json.dump(metadata, metadata_file, indent=2,
                          separators=(',', ':'))

            return metadata, first_page["numRecordings"]

    def create_datasets(self, species_list, test_size=0.35, min_quality="E", sound_types=["song"], sexes=["male", "female", "sex uncertain"], life_stages=["adult", "juvenile",
                                                                                                                                                           "hatchling or nestling", "life stage uncertain"], exclude_special_cases=True, maximum_number_of_background_species=None):
        if min_quality not in self.xc_quality_levels:
            raise ValueError("Invalid quality level for Xeno-Canto database")
        if not set(sound_types).issubset(self.xc_sound_types):
            raise ValueError("Invalid sound type for Xeno-Canto database")
        if not set(sexes).issubset(self.xc_sexes):
            raise ValueError("Invalid sex for Xeno-Canto database")
        if not set(life_stages).issubset(self.xc_life_stages):
            raise ValueError("Invalid life stage for Xeno-Canto database")

        train_frames = []
        test_frames = []
        val_frames = []

        for species_name in species_list:
            try:
                labels, _ = self.download_species_metadata(species_name)
            except Exception as e:
                print(e)
                return

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

            if (len(labels) == 0):
                raise NameError(
                    "There are no training samples for class {}".format(species_name))

            # create class labels
            labels["sound_type"] = ""
            for idx, row in labels.iterrows():
                for sound_type in sound_types:
                    if sound_type in row["type"]:
                        row["sound_type"] = sound_type
                        break
            labels["label"] = labels["gen"] + "_" + \
                labels["sp"] + "_" + labels["sound_type"]

            # select relevant columns
            labels = labels[["id", "label", "q",
                             "sound_type", "background_species"]]

            # create train, test and val splits
            train_labels, test_labels = train_test_split(
                labels, test_size=test_size, random_state=12)

            test_labels, val_labels = train_test_split(
                test_labels, test_size=test_size, random_state=12)

            train_frames.append(train_labels)
            test_frames.append(test_labels)
            val_frames.append(val_labels)

        # save label files
        training_set = pd.concat(train_frames)
        test_set = pd.concat(test_frames)
        validation_set = pd.concat(val_frames)
        training_set.to_json(os.path.join(
            self.train_dir, "train.json"), "records", indent=4)
        test_set.to_json(os.path.join(
            self.test_dir, "test.json"), "records", indent=4)
        validation_set.to_json(os.path.join(
            self.val_dir, "val.json"), "records", indent=4)

        # download audio files
        self.download_audio_files_by_id(
            self.train_audio_dir, training_set["id"], "Download training set")

        self.download_audio_files_by_id(
            self.test_audio_dir, test_set["id"], "Download test set")

        self.download_audio_files_by_id(
            self.val_audio_dir, validation_set["id"], "Download validation set")
