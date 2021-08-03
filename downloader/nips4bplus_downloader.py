import os
import pandas as pd
import shutil
import tarfile
from typing import List
import zipfile

from .downloader import Downloader
from general import logger, PathManager


class NIPS4BPlusDownloader(Downloader):
    def __init__(self, path_manager: PathManager):
        super().__init__(path_manager)

    def __del__(self):
        super().__del__

    def download_nips4bplus_dataset(self, species_list: List[str]):
        nips4bplus_annotations_url = "https://ndownloader.figshare.com/files/16334603"
        nips4bplus_annotations_folder_name = "temporal_annotations_nips4b"
        nips4b_audio_files_url = "http://sabiod.univ-tln.fr/nips4b/media/birds/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV.tar.gz"
        nips4b_audio_folder_name = "NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV"

        nips4bplus_folder = self.path.data_folder("nips4bplus", "")
        nips4bplus_folder_all = self.path.data_folder("nips4bplus_all", "")

        self.path.empty_dir(nips4bplus_folder)
        nips4bplus_audio_folder = self.path.data_folder("nips4bplus", "audio")
        nips4bplus_all_audio_folder = self.path.data_folder("nips4bplus_all", "audio")
        nips4bplus_annotations_path = os.path.join(nips4bplus_folder, "nips4bplus_annotations.zip")
        nips4bplus_audio_path = os.path.join(nips4bplus_folder, "nips4bplus_audio.tar.gz")
        extracted_nips_annotations_folder = os.path.join(nips4bplus_folder, nips4bplus_annotations_folder_name)

        nips4b_species_list = self.download_nips4b_species_list()

        logger.info("Download NIPS4BPlus label files...")
        self.download_file(nips4bplus_annotations_url, nips4bplus_annotations_path, cache_subdir="nips4bplus")

        with zipfile.ZipFile(nips4bplus_annotations_path, 'r') as zip_file:
            logger.info("Unzip NIPS4BPlus label files...")
            zip_file.extractall(nips4bplus_folder)

        nips4bplus_selected_labels = []
        nips4bplus_labels = []

        species_names = [species_name for species_name, sound_types in
                         self.parse_species_list(species_list, {})]
        selected_species = '|'.join(species_names)

        for file in os.listdir(extracted_nips_annotations_folder):
            label_file_path = os.path.join(extracted_nips_annotations_folder, file)

            def map_class_names(row):
                if row["label"] == "Unknown":
                    return "noise"
                elif row["label"] == "Human":
                    return "noise"

                class_name = nips4b_species_list[nips4b_species_list["nips4b_class_name"] == row["label"]]

                if len(class_name) != 1:
                    raise NameError(f"No unique label found for class {row['label']}")

                if class_name["Scientific_name"].item() not in species_names:
                    return "noise"
                else:
                    return class_name["class name"].item()

            if file.endswith(".csv"):
                try:
                    labels = pd.read_csv(label_file_path, names=["start", "duration", "label"])
                    labels["label"] = labels.apply(map_class_names, axis=1)
                except pd.errors.EmptyDataError:
                    labels = pd.DataFrame([0, 5, "noise"], columns=["start", "duration", "label"])
                file_id = file.lstrip("annotation_train").rstrip(".csv")

                labels["id"] = f"nips4b_birds_trainfile{file_id}"
                labels["file_name"] = f"nips4b_birds_trainfile{file_id}.wav"
                labels["start"] = labels["start"] * 1000
                labels["end"] = labels["start"] + labels["duration"] * 1000

                contains_selected_species = False
                for idx, label in labels.iterrows():
                    class_name = nips4b_species_list[nips4b_species_list["class name"] == label["label"]]

                    if label["label"] != "noise" and class_name["Scientific_name"].item() in species_names:
                        contains_selected_species = True
                if contains_selected_species:
                    nips4bplus_selected_labels.append(labels)

                labels = labels[["id", "file_name", "start", "end", "label"]]

                self.append = nips4bplus_labels.append(labels)

        nips4bplus_labels = pd.concat(nips4bplus_labels)
        nips4bplus_labels.to_csv(os.path.join(nips4bplus_folder_all, "nips4bplus_all.csv"))
        if len(nips4bplus_selected_labels) > 0:
            nips4bplus_selected_labels = pd.concat(nips4bplus_selected_labels)
            nips4bplus_selected_labels.to_csv(os.path.join(nips4bplus_folder, "nips4bplus.csv"))

        os.remove(nips4bplus_annotations_path)
        shutil.rmtree(extracted_nips_annotations_folder)

        logger.info("Download NIPS4BPlus audio files...")
        self.download_file(nips4b_audio_files_url, nips4bplus_audio_path, cache_subdir="nips4bplus")

        logger.info("Unzip NIPS4BPlus audio files...")
        tar_file = tarfile.open(nips4bplus_audio_path)
        tar_file.extractall(nips4bplus_folder)
        tar_file.close()
        extracted_nips_audio_folder = os.path.join(nips4bplus_folder, nips4b_audio_folder_name)

        for split in ["train", "test"]:
            folder_path = os.path.join(extracted_nips_audio_folder, split)
            PathManager.copytree(folder_path, nips4bplus_audio_folder)
            PathManager.copytree(folder_path, nips4bplus_all_audio_folder)

        os.remove(nips4bplus_audio_path)
        shutil.rmtree(extracted_nips_audio_folder)

        for file in os.listdir(nips4bplus_audio_folder):
            if nips4bplus_selected_labels[nips4bplus_selected_labels["file_name"] == file].empty:
                os.remove(os.path.join(nips4bplus_audio_folder, file))
        for file in os.listdir(nips4bplus_all_audio_folder):
            if nips4bplus_labels[nips4bplus_labels["file_name"] == file].empty:
                os.remove(os.path.join(nips4bplus_all_audio_folder, file))

    def download_nips4b_species_list(self):
        species_list_url = "https://ndownloader.figshare.com/files/13390469"
        nips4bplus_folder = self.path.data_folder("nips4bplus", "")
        nips4bplus_species_list = os.path.join(nips4bplus_folder, "nips4b_species_list.csv")

        self.download_file(species_list_url, nips4bplus_species_list, cache_subdir="nips4bplus")

        species_list = pd.read_csv(nips4bplus_species_list)
        species_list["nips4b_class_name"] = species_list["class name"]

        species_list["class name"] = species_list.apply(
            lambda row: row["Scientific_name"].replace(" ", "_") + "_" + row["class name"].split("_")[1] if row[
                                                                                                                "class name"] != "Empty" else "noise sample",
            axis=1)

        species_list.to_csv(nips4bplus_species_list)

        return species_list
