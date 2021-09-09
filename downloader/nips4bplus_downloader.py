import os
import shutil
import tarfile
import zipfile
from typing import List

import pandas as pd

from general import logger, PathManager
from .downloader import Downloader


class NIPS4BPlusDownloader(Downloader):
    """
    Downloads NIPS4BPlus dataset.

    see Veronica Morfi et al. “NIPS4Bplus: a richly annotated birdsong audio dataset.” In: PeerJ Computer Science 5.e223
    (Oct. 7, 2019), pp. 1–12. ISSN: 2376-5992. DOI: 10.7717/peerj-cs.223.
    """

    nips4bplus_annotations_url = "https://ndownloader.figshare.com/files/16334603"
    nips4bplus_annotations_folder_name = "temporal_annotations_nips4b"
    nips4b_audio_files_url = "http://sabiod.univ-tln.fr/nips4b/media/birds/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV.tar.gz"
    nips4b_audio_folder_name = "NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV"
    species_list_url = "https://ndownloader.figshare.com/files/13390469"
    nips4bplus_sound_types_to_xc_sound_types = {"call": "call", "drum": "drumming", "song": "song"}

    def __init__(self, path_manager: PathManager) -> None:
        """

        Args:
            path_manager: PathManager object that manages the output directory to be used for storing the
                downloaded datasets.
        """

        super().__init__(path_manager)

        self.nips4bplus_folder = self.path.data_folder("nips4bplus")
        self.nips4bplus_folder_all = self.path.data_folder("nips4bplus_all")
        self.extracted_nips_annotations_folder = os.path.join(self.nips4bplus_folder,
                                                              NIPS4BPlusDownloader.nips4bplus_annotations_folder_name)
        self.extracted_nips_audio_folder = os.path.join(
            self.nips4bplus_folder, NIPS4BPlusDownloader.nips4b_audio_folder_name)

    def __download_nips4b_audio_files(self) -> None:
        """
        Downloads audio files of NIPS4B dataset.

        Returns:
            None
        """

        nips4bplus_audio_path = os.path.join(self.nips4bplus_folder, "nips4bplus_audio.tar.gz")

        logger.info("Download NIPS4BPlus audio files...")
        self.download_file(NIPS4BPlusDownloader.nips4b_audio_files_url, nips4bplus_audio_path,
                           cache_subdir="nips4bplus")

        logger.info("Unzip NIPS4BPlus audio files...")
        tar_file = tarfile.open(nips4bplus_audio_path)
        tar_file.extractall(self.nips4bplus_folder)
        tar_file.close()

        os.remove(nips4bplus_audio_path)

    def __download_nips4b_plus_annotations(self) -> None:
        """
        Downloads temporal annotations for NIPS4B audio files from NIPS4BPlus dataset.

        Returns:
            None
        """

        nips4bplus_annotations_path = os.path.join(self.nips4bplus_folder, "nips4bplus_annotations.zip")

        logger.info("Download NIPS4BPlus label files...")
        self.download_file(NIPS4BPlusDownloader.nips4bplus_annotations_url, nips4bplus_annotations_path,
                           cache_subdir="nips4bplus")

        with zipfile.ZipFile(nips4bplus_annotations_path, 'r') as zip_file:
            logger.info("Unzip NIPS4BPlus label files...")
            zip_file.extractall(self.nips4bplus_folder)

        os.remove(nips4bplus_annotations_path)

    def download_nips4b_species_list(self) -> pd.DataFrame:
        """
        Downloads list of categories / species of the NIPS4B dataset.

        Returns:
            Pandas Dataframe that contains the columns "nips4b_class_name", "class name", "Scientific_name", and "sound_type".
        """

        nips4bplus_species_list = os.path.join(self.path.data_folder("nips4bplus", ""), "nips4b_species_list.csv")

        self.download_file(NIPS4BPlusDownloader.species_list_url, nips4bplus_species_list, cache_subdir="nips4bplus")

        species_list = pd.read_csv(nips4bplus_species_list)
        species_list["nips4b_class_name"] = species_list["class name"]
        species_list["sound_type"] = species_list["nips4b_class_name"].apply(
            lambda class_name: class_name.split("_")[1] if len(class_name.split("_")) > 1 else "")
        species_list["sound_type"] = species_list["sound_type"].apply(
            lambda sound_type: NIPS4BPlusDownloader.nips4bplus_sound_types_to_xc_sound_types[
                sound_type] if sound_type in NIPS4BPlusDownloader.nips4bplus_sound_types_to_xc_sound_types else "")

        def get_class_name(row):
            if row["class name"] != "Empty":
                return row["Scientific_name"].replace(" ", "_") + "_" + row["class name"].split("_")[1]
            else:
                return "noise sample"

        species_list["class name"] = species_list.apply(get_class_name, axis=1)

        species_list.to_csv(nips4bplus_species_list)

        return species_list

    def __create_label_file(self, species_list: List[str]) -> None:
        """
        Creates label file for the NIPS4BPlus dataset using the categories from the provided species list.
        The list has to be in the format ["species name, sound type name 1, sound type name 2, ...", "..."].

        Args:
            species_list: List of species and sound types in the above mentioned  format.

        Returns:
            None
        """

        nips4bplus_audio_folder = self.path.data_folder("nips4bplus", "audio")
        nips4bplus_all_audio_folder = self.path.data_folder("nips4bplus_all", "audio")

        nips4b_species_list = self.download_nips4b_species_list()

        nips4bplus_selected_labels = []
        nips4bplus_labels = []

        species_to_sound_types = self.parse_species_list(species_list, {"song", "call"})

        for file in os.listdir(self.extracted_nips_annotations_folder):
            label_file_path = os.path.join(self.extracted_nips_annotations_folder, file)

            def map_class_names(row):
                if row["label"] in ('Unknown', 'Human'):
                    return "noise"

                nips4b_class_name = nips4b_species_list[nips4b_species_list["nips4b_class_name"] == row["label"]]
                scientific_n = nips4b_class_name["Scientific_name"].item()
                sound_t = nips4b_class_name["sound_type"].item()

                if len(nips4b_class_name) != 1:
                    raise NameError(f"No unique label found for class {row['label']}")

                if scientific_n not in species_to_sound_types or sound_t not in species_to_sound_types[scientific_n]:
                    return "noise"
                else:
                    return nips4b_class_name["class name"].item()

            if file.endswith(".csv"):
                try:
                    labels = pd.read_csv(label_file_path, names=["start", "duration", "label"])
                    labels["label"] = labels.apply(map_class_names, axis=1)
                except pd.errors.EmptyDataError:
                    labels = pd.DataFrame([0, 5, "noise"], columns=["start", "duration", "label"])

                file_id = file.lstrip("annotation_train").rstrip(".csv")

                labels["id"] = f"nips4b_birds_trainfile{file_id}"
                labels["file_path"] = f"nips4b_birds_trainfile{file_id}.wav"
                labels["start"] = labels["start"] * 1000
                labels["end"] = labels["start"] + labels["duration"] * 1000

                contains_selected_species = False
                for idx, label in labels.iterrows():
                    class_name = nips4b_species_list[nips4b_species_list["class name"] == label["label"]]

                    if label["label"] != "noise" and class_name["Scientific_name"].item() in species_to_sound_types:
                        contains_selected_species = True
                if contains_selected_species:
                    nips4bplus_selected_labels.append(labels)

                labels = labels[["id", "file_path", "start", "end", "label"]]

                self.append = nips4bplus_labels.append(labels)

        nips4bplus_labels = pd.concat(nips4bplus_labels)
        self.save_label_file(nips4bplus_labels, "nips4bplus_all")
        if len(nips4bplus_selected_labels) > 0:
            nips4bplus_selected_labels = pd.concat(nips4bplus_selected_labels)
        else:
            nips4bplus_selected_labels = pd.DataFrame(columns=["id", "file_path", "label", "start", "end"])

        self.save_label_file(nips4bplus_selected_labels, "nips4bplus")

        for dataset in ["train", "test"]:
            folder_path = os.path.join(self.extracted_nips_audio_folder, dataset)
            PathManager.copytree(folder_path, nips4bplus_audio_folder)
            PathManager.copytree(folder_path, nips4bplus_all_audio_folder)

        # remove audio files without labels
        for file in os.listdir(nips4bplus_audio_folder):
            if nips4bplus_selected_labels[nips4bplus_selected_labels["file_path"] == file].empty:
                os.remove(os.path.join(nips4bplus_audio_folder, file))
        for file in os.listdir(nips4bplus_all_audio_folder):
            if nips4bplus_labels[nips4bplus_labels["file_path"] == file].empty:
                os.remove(os.path.join(nips4bplus_all_audio_folder, file))

    def download_nips4bplus_dataset(self, species_list: List[str]) -> None:
        """
        Downloads the whole NIPS4BPlus dataset and creates labels using the categories from the provided species list.
        The list has to be in the format ["species name, sound type name 1, sound type name 2, ...", "..."].

        Args:
            species_list: List of species and sound types in the above mentioned  format.

        Returns:
            None
        """

        self.path.empty_dir(self.nips4bplus_folder)

        self.__download_nips4b_audio_files()
        self.__download_nips4b_plus_annotations()
        self.__create_label_file(species_list)

        shutil.rmtree(self.extracted_nips_annotations_folder)
        shutil.rmtree(self.extracted_nips_audio_folder)
