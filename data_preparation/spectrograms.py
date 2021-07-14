import librosa
import math
from multiprocessing.pool import ThreadPool
import numpy
import pandas as pd
import os
from scipy import ndimage
import shutil
from skimage import io
from typing import List, Optional
import warnings

from data_preparation.filepaths import PathManager
from general.logging import logger, ProgressBar

warnings.filterwarnings('ignore')


class SpectrogramCreator:
    def __init__(self, chunk_length: int, audio_path_manager: PathManager,
                 spectrogram_path_manager: Optional[PathManager] = None, include_noise_samples: bool = True):
        # parameters for spectorgram creation
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

        self.index_cached_spectrograms()

    def __del__(self):
        if self.spectrogram_path.is_pipeline_run:
            self.spectrogram_path.empty_dir(self.spectrogram_path.cache_dir)


    def scale_min_max(self, x, min_value: float = 0.0, max_value: float = 1.0, min_value_source: Optional[float] = None,
                      max_value_source: Optional[float] = None):
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

    def filter_noise(self, spectrogram: numpy.ndarray):
        # normalize spectrogram to [0, 1]
        spectrogram = self.scale_min_max(
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
        # code adapted from https://github.com/kahst/BirdCLEF2017/blob/f485a3f9083b35bdd7a276dcd1c14da3a9568d85/birdCLEF_spec.py#L120

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

    def contains_signal(self, spectrogram: numpy.ndarray, signal_threshold: int = 3, noise_threshold: int = 1):
        filtered_spectrogram = self.filter_noise(spectrogram)

        # count rows with signal
        row_max = numpy.max(filtered_spectrogram, axis=1)

        # apply binary dilation to array with max values
        row_max = ndimage.morphology.binary_dilation(
            row_max, iterations=2).astype(row_max.dtype)

        rows_with_signal = row_max.sum()

        return rows_with_signal >= signal_threshold, rows_with_signal < noise_threshold

    def save_spectrogram(self, target_file: str, spectrogram: numpy.ndarray):
        # scale amplitude values to range [0, 255]
        # invert image so that black represents higher amplitudes
        img = self.scale_min_max(
            spectrogram, 255, 0, 0.0, 1.0).astype(numpy.uint8)

        # put low frequencies at the bottom of the image
        img = numpy.flip(img, axis=0)

        io.imsave(target_file, img)

        # copy spectrogram to cache
        cached_file_path = self.spectrogram_path.cached_file_path(
            "spectrograms", target_file, chunk_length=self.chunk_length)
        shutil.copy(target_file, cached_file_path)

        if self.spectrogram_path.is_pipeline_run:
            self.spectrogram_path.copy_file_to_gcs_cache(cached_file_path, "spectrograms",
                                                         chunk_length=self.chunk_length)

    def index_cached_spectrograms(self):
        cache_path = self.spectrogram_path.cache("spectrograms", chunk_length=self.chunk_length)

        self.cached_spectrograms = {
            "with noise": {},
            "without noise": {}
        }
        self.cached_spectrograms_without_noise = {}

        for cached_spectrogram in os.listdir(cache_path):
            file_id = cached_spectrogram.split("-")[0].rstrip(".png")
            if not file_id in self.cached_spectrograms["with noise"]:
                self.cached_spectrograms["with noise"][file_id] = []
            if not file_id in self.cached_spectrograms["without noise"]:
                self.cached_spectrograms["without noise"][file_id] = []
            spectrogram_path = os.path.join(
                cache_path, cached_spectrogram)
            self.cached_spectrograms["with noise"][file_id].append(spectrogram_path)

            if not "noise" in cached_spectrogram:
                self.cached_spectrograms["without noise"][file_id].append(spectrogram_path)

    def get_cached_spectrograms(self, audio_file: str):
        audio_file_id = os.path.splitext(os.path.basename(audio_file))[0]

        if self.include_noise_samples and audio_file_id in self.cached_spectrograms["with noise"]:
            return self.cached_spectrograms["with noise"][audio_file_id]
        elif not self.include_noise_samples and audio_file_id in self.cached_spectrograms["without noise"]:
            return self.cached_spectrograms["without noise"][audio_file_id]

        return []

    def create_spectrograms_from_file(self, audio_file: str, target_dir: str, signal_threshold: int, noise_threshold: int):
        cached_spectrograms_for_current_file = self.get_cached_spectrograms(
            audio_file)
        if len(cached_spectrograms_for_current_file) > 0:
            logger.verbose("cached spectrograms for %s", audio_file)
            for file in cached_spectrograms_for_current_file:
                shutil.copy(file, target_dir)

        else:
            logger.verbose("process %s", audio_file)
            # load audio file
            amplitudes, sr = librosa.load(audio_file, sr=self.sampling_rate)

            audio_length = amplitudes.shape[0]

            number_of_chunks = math.floor(
                audio_length / self.samples_per_chunk)

            # split audio file in chunks and create one spectrogram per chunk
            for i in range(number_of_chunks):
                # get samples of current chunk
                chunk = amplitudes[i *
                                   self.samples_per_chunk:(i + 1) * self.samples_per_chunk]

                # apply short time fourier transformation to extract frequency information from amplitude data
                mel_spectrogram = librosa.feature.melspectrogram(chunk, sr=self.sampling_rate,
                                                                 hop_length=self.hop_length, n_fft=self.n_fft,
                                                                 win_length=self.window_length, n_mels=112,
                                                                 fmin=self.fmin, fmax=self.fmax)

                # convert power spectrogram to dB-scaled spectrogram
                mel_spectrogram_db = librosa.power_to_db(
                    mel_spectrogram, ref=numpy.max)

                file_name = os.path.splitext(os.path.basename(audio_file))[0]
                target_file = os.path.join(
                    target_dir, "{}-{}.png".format(file_name, i))

                contains_signal, is_noise = self.contains_signal(
                    mel_spectrogram_db, signal_threshold=signal_threshold, noise_threshold=noise_threshold)

                if contains_signal:
                    self.save_spectrogram(target_file, mel_spectrogram_db)
                elif is_noise and self.include_noise_samples:
                    target_file = os.path.join(
                        target_dir, "{}-{}_noise.png".format(file_name, i))
                    self.save_spectrogram(target_file, mel_spectrogram_db)

    def create_spectrograms_from_dir(self, audio_dir: str, target_dir: str, signal_threshold: int, noise_threshold: int, desc: Optional[str] = None, spectrogram_creation_threads=25):
        # clean up target dir
        PathManager.empty_dir(target_dir)

        audio_file_names = os.listdir(audio_dir)

        progress_bar = ProgressBar(
            total=len(audio_file_names), desc="Create spectrograms for {}".format(desc), position=0,
            is_pipeline_run=self.spectrogram_path.is_pipeline_run)

        pool = ThreadPool(spectrogram_creation_threads)

        def spectrogram_task(file_name):
            if file_name.endswith(".mp3") or file_name.endswith(".wav"):
                audio_path = os.path.join(audio_dir, file_name)
                self.create_spectrograms_from_file(audio_path, target_dir, signal_threshold, noise_threshold)

        for _ in pool.imap_unordered(lambda file_name: spectrogram_task(file_name), audio_file_names):
            progress_bar.update(1)

    def create_spectrograms_for_splits(self, splits: Optional[List[str]] = None, signal_threshold: int = 3, noise_threshold: int = 1, clear_spectrogram_cache: bool = False, ):
        if splits is None:
            splits = ["train", "val", "test", "nips4bplus", "nips4bplus_all"]

        if clear_spectrogram_cache:
            self.spectrogram_path.clear_cache("spectrograms", chunk_length=self.chunk_length)

        for split in splits:
            spectrogram_dir = self.spectrogram_path.data_folder(
                split, "spectrograms", chunk_length=self.chunk_length)
            audio_dir = self.audio_path.data_folder(split, "audio")
            audio_label_file = self.audio_path.audio_label_file(split)
            PathManager.ensure_dir(spectrogram_dir)
            self.create_spectrograms_from_dir(
                audio_dir, spectrogram_dir, signal_threshold, noise_threshold, f"{split} set")
            self.create_spectrogram_labels(split)

    def create_spectrogram_labels(self, split: str):
        labels = pd.read_csv(self.audio_path.audio_label_file(split))
        spectrogram_dir = self.spectrogram_path.data_folder(split, "spectrograms", chunk_length=self.chunk_length)

        spectrogram_labels = []

        for file in os.listdir(spectrogram_dir):
            if file.endswith(".png"):
                file_id = file.split("-")[0]

                file_number = file.split("-")[1]
                file_number = file_number.split("_")[0]
                file_number = file_number.split(".")[0]
                file_number = int(file_number)

                start = file_number * self.chunk_length
                end = (file_number+1) * self.chunk_length
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
                    if start <= label["start"] and label["start"] < end or start <= label["end"] and label["end"] < end:
                        final_label[label["label"]] = 1
                        final_label["id"] = file_id
                        final_label["file_name"] = file
                        found_match = True

                if not found_match:
                    final_label["noise"] = 1
                    final_label["id"] = file_id
                    final_label["file_name"] = file

                spectrogram_labels.append(final_label)

        if len(spectrogram_labels) > 0:
            spectrogram_labels = pd.DataFrame(spectrogram_labels).sort_values(
                by=['file_name']).fillna(0)

            label_file = self.spectrogram_path.spectrogram_label_file(split, chunk_length=self.chunk_length)
            spectrogram_labels.to_csv(label_file)
        else:
            raise NameError("No spectrograms found")
