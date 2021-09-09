import math
import os
import shutil
import warnings
from typing import List, Optional, Tuple

import joblib
import librosa
import numpy
import pandas as pd
from scipy import ndimage
from skimage import io

from general.file_manager import FileManager
from general.logging import logger, ProgressBar

warnings.filterwarnings('ignore')


class SpectrogramCreator:
    """
    Creates spectrogram images from audio files.
    """

    @staticmethod
    def __scale_min_max(x: numpy.ndarray, min_value: float = 0.0, max_value: float = 1.0,
                        min_value_source: Optional[float] = None,
                        max_value_source: Optional[float] = None) -> numpy.ndarray:
        """
        Scales the values of the given array to the interval [min_value, max_value].

        Args:
            x: A numpy ndarray whose values are to be scaled.
            min_value: Lower bound of the target interval.
            max_value: Upper bound of the target interval.
            min_value_source: Lower bound of the original interval.
            max_value_source: Upper bound of the original interval.

        Returns:
            Numpy ndarray with scaled values.
        """

        invert_image = False
        if not min_value_source:
            min_value_source = x.min()
        if not max_value_source:
            max_value_source = x.max()
        if min_value > max_value:
            # if min value is greater than max value, the image is inverted
            invert_image = True
            min_tmp = min_value
            min_value = max_value
            max_value = min_tmp

        # scale values to target interval
        x_std = (x - min_value_source) / (max_value_source - min_value_source)
        x_scaled = x_std * (max_value - min_value) + min_value

        if invert_image:
            x_scaled = max_value - x_scaled

        return x_scaled

    def __init__(self, chunk_length: int, audio_path_manager: FileManager,
                 spectrogram_path_manager: Optional[FileManager] = None, include_noise_samples: bool = True) -> None:
        """

        Args:
            chunk_length: Length of the chunks into which the audio recordings are to be split (in milliseconds). Per
                chunk one spectrogram is created.
            audio_path_manager: FileManager object that manages the input directory containing the audio files from
                which the spectrograms are to be created.
            spectrogram_path_manager: FileManager object that manages the output directory to be used for storing the
                created spectrograms.
            include_noise_samples: Whether spectrograms that are classified as "noise" during noise filtering should be
                included in the spectrogram dataset.
        """

        # parameters for spectrogram creation
        self.chunk_length = chunk_length  # chunk length in milliseconds
        self.sampling_rate = 44100  # number of samples per second
        self.window_length = 1024
        self.n_fft = 1024
        self.hop_length = 196
        self.fmin = 500  # minimum frequency
        self.fmax = 15000  # maximum frequency

        self.samples_per_chunk = math.floor(
            (self.sampling_rate * chunk_length) / 1000)

        self.include_noise_samples = include_noise_samples

        if not spectrogram_path_manager:
            spectrogram_path_manager = audio_path_manager
        self.audio_path = audio_path_manager
        self.spectrogram_path = spectrogram_path_manager

        if self.spectrogram_path.is_pipeline_run:
            self.spectrogram_path.copy_cache_from_gcs("spectrograms", chunk_length=self.chunk_length)

        self.__index_cached_spectrograms()

    def __filter_noise(self, spectrogram: numpy.ndarray) -> numpy.ndarray:
        """
        Applies multiple image filters to a spectrogram image to extract its signal part.
        In the resulting spectrogram image, black pixels represent signal parts and white pixels represent noise parts.

        Args:
            spectrogram: Input spectrogram image.

        Returns:
            Filtered spectrogram image.
        """

        # normalize spectrogram to [0, 1]
        spectrogram = self.__scale_min_max(
            spectrogram.astype(numpy.double), 0.0, 1.0).astype(numpy.double)

        # apply median blur with kernel size 5
        # each pixel is replaced by the median gray value of the kernel
        filtered_spectrogram = ndimage.median_filter(spectrogram, size=5)

        # apply median filtering
        # pixels that are 1.5 times larger than the row and the column median are set to black
        # all other pixels are set to white
        col_median = numpy.median(filtered_spectrogram, axis=0, keepdims=True)
        row_median = numpy.median(filtered_spectrogram, axis=1, keepdims=True)

        filtered_spectrogram[filtered_spectrogram < row_median * 1.5] = 0.0
        filtered_spectrogram[filtered_spectrogram < col_median * 1.5] = 0.0
        filtered_spectrogram[filtered_spectrogram > 0] = 1.0

        # spot removal: filter out isolated black pixels
        # code adapted from
        # https://github.com/kahst/BirdCLEF2017/blob/f485a3f9083b35bdd7a276dcd1c14da3a9568d85/birdCLEF_spec.py#L120

        # create matrix that indicates for each pixel to which region it belongs
        # a region is a connected area of black pixels
        struct = numpy.ones((3, 3))
        region_labels, num_regions = ndimage.label(
            filtered_spectrogram, structure=struct)

        # calculate size (number of black pixels) of each region
        region_sizes = numpy.array(ndimage.sum(
            filtered_spectrogram, region_labels, range(num_regions + 1)))

        # set isolated black pixels to zero
        region_mask = (region_sizes == 1)
        filtered_spectrogram[region_mask[region_labels]] = 0

        # apply morphology closing
        struct = numpy.ones((5, 5))
        filtered_spectrogram = ndimage.morphology.binary_closing(
            filtered_spectrogram, structure=struct).astype(numpy.int)

        return filtered_spectrogram

    def __contains_signal(self, spectrogram: numpy.ndarray, signal_threshold: int = 3, noise_threshold: int = 1
                          ) -> Tuple[bool, bool]:
        """
        Counts the number of image rows containing signal pixels and compares them to a signal or a noise threshold to
        determine whether the spectrogram contains substantial signal parts or only noise.

        Args:
            spectrogram: Input spectrogram image.
            signal_threshold: Threshold used to identify "signal" spectrograms. If the number of image rows containing
                signal pixels is equal or above this threshold, the spectrogram is classified as containing substantial
                signal parts.
            noise_threshold: Threshold used to identify "noise" spectrograms. If the number of image rows containing
                signal pixels is equal or below this threshold, the spectrogram is classified as containing only noise.

        Returns:
            Tuple of two Booleans. The first value is True, if the spectrogram image is classified as "signal"
            spectrogram. The second value is True, if the spectrogram image is classified as "noise". If the noise
            threshold is smaller than the signal theshold, both values may be False. In this case it is uncertain
            whether the spectrogram contains signal parts.
        """

        assert noise_threshold <= signal_threshold

        filtered_spectrogram = self.__filter_noise(spectrogram)

        row_max = numpy.max(filtered_spectrogram, axis=1)

        # apply binary dilation to array with max values
        # see https://github.com/kahst/BirdCLEF2017/blob/f485a3f9083b35bdd7a276dcd1c14da3a9568d85/birdCLEF_spec.py#L120
        row_max = ndimage.morphology.binary_dilation(
            row_max, iterations=2).astype(row_max.dtype)

        # count rows with signal
        rows_with_signal = row_max.sum()

        return rows_with_signal >= signal_threshold, rows_with_signal < noise_threshold

    def __save_spectrogram(self, target_file: str, spectrogram: numpy.ndarray) -> None:
        """
        Saves spectrogram to an image file.

        Args:
            target_file: Absolute path of the image file.
            spectrogram: Numpy ndarray containing the pixel values of the spectrogram.

        Returns:
            None
        """

        # scale amplitude values to range [0, 255]
        # invert image so that black represents higher amplitudes
        img = self.__scale_min_max(spectrogram, 255, 0, 0.0, 1.0).astype(numpy.uint8)

        # put low frequencies at the bottom of the image
        img = numpy.flip(img, axis=0)

        io.imsave(target_file, img)

        # copy spectrogram to cache
        cached_file_path = self.spectrogram_path.cached_file_path(
            "spectrograms", target_file, chunk_length=self.chunk_length)
        shutil.copy(target_file, cached_file_path)

    def __index_cached_spectrograms(self) -> None:
        """
        Creates an index containing the audio files whose spectrograms are already in the cache.

        Returns:
            None
        """

        cache_path = self.spectrogram_path.cache("spectrograms", chunk_length=self.chunk_length)

        self.cached_spectrograms = {
            "with noise": {},
            "without noise": {}
        }
        self.cached_spectrograms_without_noise = {}

        for cached_spectrogram in os.listdir(cache_path):
            file_id = cached_spectrogram.split("-")[0].rstrip(".png")
            if file_id not in self.cached_spectrograms["with noise"]:
                self.cached_spectrograms["with noise"][file_id] = []
            if file_id not in self.cached_spectrograms["without noise"]:
                self.cached_spectrograms["without noise"][file_id] = []
            spectrogram_path = os.path.join(
                cache_path, cached_spectrogram)
            self.cached_spectrograms["with noise"][file_id].append(spectrogram_path)

            if "noise" not in cached_spectrogram:
                self.cached_spectrograms["without noise"][file_id].append(spectrogram_path)

    def __get_cached_spectrograms(self, audio_file: str) -> list:
        """
        Creates a list of the paths of the spectrograms for an audio file that are already in cache.

        Args:
            audio_file: Path of the audio file.

        Returns:
            List of paths of cached spectrograms.
        """

        audio_file_id = os.path.splitext(os.path.basename(audio_file))[0]

        if self.include_noise_samples and audio_file_id in self.cached_spectrograms["with noise"]:
            return self.cached_spectrograms["with noise"][audio_file_id]
        elif not self.include_noise_samples and audio_file_id in self.cached_spectrograms["without noise"]:
            return self.cached_spectrograms["without noise"][audio_file_id]

        return []

    def __create_spectrograms_from_file(self, audio_file: str, target_dir: str, signal_threshold: int,
                                        noise_threshold: int) -> List[str]:
        """
        Splits audio file into several chunks and creates one spectrogram image per chunk.

        Args:
            audio_file: Path of the input audio file.
            target_dir: Path of the directory in which the spectrograms are to be stored.
            signal_threshold: Threshold for identifying "signal" spectrograms (see documentation of __contains_signal).
            noise_threshold: Threshold for identifying "noise" spectrograms (see documentation of __contains_signal).

        Returns:
            List of paths of the created spectrogram files.
        """

        cached_spectrograms_for_current_file = self.__get_cached_spectrograms(audio_file)
        if len(cached_spectrograms_for_current_file) > 0:
            logger.verbose("cached spectrograms for %s", audio_file)
            for file in cached_spectrograms_for_current_file:
                shutil.copy(file, target_dir)

            return []

        else:
            logger.verbose("process %s", audio_file)
            # load audio file
            try:
                amplitudes, sr = librosa.load(audio_file, sr=self.sampling_rate)

                audio_length = amplitudes.shape[0]

                number_of_chunks = math.floor(audio_length / self.samples_per_chunk)

                spectrogram_paths = []

                # split audio file in chunks and create one spectrogram per chunk
                for i in range(number_of_chunks):
                    # get samples of current chunk
                    chunk = amplitudes[i * self.samples_per_chunk:(i + 1) * self.samples_per_chunk]

                    # apply short time fourier transformation to extract frequency information from amplitude data
                    mel_spectrogram = librosa.feature.melspectrogram(chunk, sr=self.sampling_rate,
                                                                     hop_length=self.hop_length, n_fft=self.n_fft,
                                                                     win_length=self.window_length, n_mels=112,
                                                                     fmin=self.fmin, fmax=self.fmax)

                    # convert power spectrogram to dB-scaled spectrogram
                    mel_spectrogram_db = librosa.power_to_db(
                        mel_spectrogram, ref=numpy.max)

                    file_name = os.path.splitext(os.path.basename(audio_file))[0]
                    target_file = os.path.join(target_dir, "{}-{}.png".format(file_name, i))

                    contains_signal, is_noise = self.__contains_signal(
                        mel_spectrogram_db, signal_threshold=signal_threshold, noise_threshold=noise_threshold)

                    if contains_signal:
                        spectrogram_paths.append(target_file)
                        self.__save_spectrogram(target_file, mel_spectrogram_db)
                    elif is_noise and self.include_noise_samples:
                        target_file = os.path.join(
                            target_dir, "{}-{}_noise.png".format(file_name, i))
                        spectrogram_paths.append(target_file)
                        self.__save_spectrogram(target_file, mel_spectrogram_db)

                return spectrogram_paths
            except Exception:
                logger.info("Could not process %s", audio_file)

    def __create_spectrograms_from_dir(self, audio_dir: str, target_dir: str, signal_threshold: int,
                                       noise_threshold: int, desc: Optional[str] = None, spectrogram_creation_threads=5
                                       ) -> None:
        """
        Creates spectrograms for all audio files in a directory.

        Args:
            audio_dir: Input directory containing audio files.
            target_dir: Path of the directory in which the spectrograms are to be stored.
            signal_threshold: Threshold for identifying "signal" spectrograms (see documentation of __contains_signal).
            noise_threshold: Threshold for identifying "noise" spectrograms (see documentation of __contains_signal).
            desc: Descriptive name of the input directory.
            spectrogram_creation_threads: Number of threads to be used for parallelization of spectrogram creation.

        Returns:
            None
        """
        # clean up target dir
        FileManager.empty_dir(target_dir)

        audio_file_names = os.listdir(audio_dir)

        progress_bar = ProgressBar(
            total=len(audio_file_names), desc="Create spectrograms for {}".format(desc), position=0,
            is_pipeline_run=self.spectrogram_path.is_pipeline_run)

        def spectrogram_task(file_name: str) -> List[str]:
            if file_name.endswith(".mp3") or file_name.endswith(".wav"):
                audio_path = os.path.join(audio_dir, file_name)
                return self.__create_spectrograms_from_file(audio_path, target_dir, signal_threshold, noise_threshold)
            return []

        if spectrogram_creation_threads <= 1:
            for file_name in audio_file_names:
                spectrogram_task(file_name)
                progress_bar.update(1)
        else:
            batch_size = 50 if self.spectrogram_path.is_pipeline_run else 20
            audio_file_names_batches = [audio_file_names[x:x + batch_size] for x in
                                        range(0, len(audio_file_names), batch_size)]
            for audio_file_names in audio_file_names_batches:
                jobs = [joblib.delayed(spectrogram_task)(file_name) for file_name in audio_file_names]
                spectrogram_paths = joblib.Parallel(n_jobs=spectrogram_creation_threads)(jobs)
                spectrogram_paths = [spectrogram_path for sublist in spectrogram_paths for spectrogram_path in sublist]

                if self.spectrogram_path.is_pipeline_run:
                    self.spectrogram_path.copy_file_to_gcs_cache(spectrogram_paths, "spectrograms",
                                                                 chunk_length=self.chunk_length)

                progress_bar.update(len(audio_file_names))

    def create_spectrograms_for_datasets(self, datasets: Optional[List[str]] = None, signal_threshold: int = 3,
                                         noise_threshold: int = 1, clear_spectrogram_cache: bool = False) -> None:
        """
        Creates spectrograms for all audio files of a dataset.

        Args:
            datasets: List of dataset names (e.g., train, val, or test).
            signal_threshold: Threshold for identifying "signal" spectrograms (see documentation of __contains_signal).
            noise_threshold: Threshold for identifying "noise" spectrograms (see documentation of __contains_signal).
            clear_spectrogram_cache: Whether the spectrogram cache should be flushed before creating the spectrograms.

        Returns:
            None
        """

        if datasets is None:
            datasets = ["train", "val", "test", "nips4bplus", "nips4bplus_all"]

        if clear_spectrogram_cache:
            self.spectrogram_path.clear_cache("spectrograms", chunk_length=self.chunk_length)
            self.__index_cached_spectrograms()

        for dataset in datasets:
            spectrogram_dir = self.spectrogram_path.data_folder(dataset, "spectrograms")
            audio_dir = self.audio_path.data_folder(dataset, "audio")
            os.makedirs(spectrogram_dir, exist_ok=True)
            self.__create_spectrograms_from_dir(
                audio_dir, spectrogram_dir, signal_threshold, noise_threshold, f"{dataset} set")
            self.__create_spectrogram_labels(dataset)

    def __create_spectrogram_labels(self, dataset: str) -> None:
        """
        Creates a spectrogram label file for a given dataset.

        Args:
            dataset: Name of a dataset (e.g., train, val, or test).

        Returns:
            None
        """
        labels = pd.read_csv(self.audio_path.label_file(dataset, "audio"))
        spectrogram_dir = self.spectrogram_path.data_folder(dataset, "spectrograms")

        spectrogram_labels = []

        for file in os.listdir(spectrogram_dir):
            if file.endswith(".png"):
                file_id = file.split("-")[0]

                file_number = file.split("-")[1]
                file_number = file_number.split("_")[0]
                file_number = file_number.split(".")[0]
                file_number = int(file_number)

                start = file_number * self.chunk_length
                end = (file_number + 1) * self.chunk_length
                matching_labels = labels[labels["id"].astype(str) == file_id]

                if file.endswith("noise.png"):
                    matching_labels["label"] = "noise"
                    matching_labels["sound_type"] = "noise"

                if len(matching_labels) < 1:
                    raise NameError(
                        "No matching labels found for file with id {}".format(file_id))

                final_label = {}

                found_match = False

                for idx, label in matching_labels.iterrows():
                    if start <= label["start"] < end \
                            or start <= label["end"] < end \
                            or label["start"] <= start \
                            and end <= label["end"]:
                        final_label[label["label"]] = 1
                        final_label["id"] = file_id
                        final_label["file_path"] = file
                        found_match = True

                if not found_match:
                    if dataset in ["train", "val", "test"]:
                        logger.info("Unlabeled spectrograms for file %s", file)
                    else:
                        final_label["noise"] = 1
                        final_label["id"] = file_id
                        final_label["file_path"] = file

                spectrogram_labels.append(final_label)

        if len(spectrogram_labels) > 0:
            spectrogram_labels = pd.DataFrame(spectrogram_labels).sort_values(
                by=['file_path']).fillna(0)
            spectrogram_labels = spectrogram_labels[spectrogram_labels["file_path"] != 0]

            label_file = self.spectrogram_path.label_file(dataset, "spectrograms")
            spectrogram_labels.to_csv(label_file)
        else:
            raise NameError("No spectrograms found")
