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

    def create_datasets(self, species_list, min_quality="E", test_size=0.3, include_calls=False):
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

            print(labels)

            # filter samples by soundtype
            labels = labels[labels["type"].str.contains("song")]

            # create class labels
            labels["label"] = labels["gen"] + "_" + labels["sp"]

            # select relevant columns
            labels = labels[["id", "label", "q"]]

            # create train, test and val splits
            train_labels, test_labels = train_test_split(
                labels, test_size=0.35, random_state=12)

            test_labels, val_labels = train_test_split(
                test_labels, test_size=0.35, random_state=12)

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
