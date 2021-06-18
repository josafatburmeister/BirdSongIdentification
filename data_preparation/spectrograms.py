import librosa
import math
import numpy
import pandas as pd
import os
from scipy import ndimage
import shutil
from skimage import io
import warnings

from data_preparation.filepaths import PathManager
from general.logging import ProgressBar

warnings.filterwarnings('ignore')


class SpectrogramCreator:
    def __init__(self, chunk_length, audio_path_manager, spectrogram_path_manager=None):
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

        if not spectrogram_path_manager:
            spectrogram_path_manager = audio_path_manager
        self.audio_path = audio_path_manager
        self.spectrogram_path = spectrogram_path_manager

        if self.spectrogram_path.is_pipeline_run:
            self.spectrogram_path.copy_cache_from_gcs("spectrograms")

    def __del__(self):
        if self.spectrogram_path.is_pipeline_run:
            self.spectrogram_path.copy_cache_to_gcs("spectrograms")

    def scale_min_max(self, x, min=0.0, max=1.0, min_source=None, max_source=None):
        invert_image = False
        if not min_source:
            min_source = x.min()
        if not max_source:
            max_source = x.max()
        if min > max:
            # if min value is greater than max value, the image is inverted
            invert_image = True
            min_tmp = min
            min = max
            max = min_tmp

        # scale values to target interval
        x_std = (x - min_source) / (max_source - min_source)
        x_scaled = x_std * (max - min) + min

        if invert_image:
            x_scaled = max - x_scaled

        return x_scaled

    def filter_noise(self, spectrogram):
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

    def contains_signal(self, spectrogram, signal_threshold=16, noise_threshold=3):
        filtered_spectrogram = self.filter_noise(spectrogram)

        # count rows with signal
        row_max = numpy.max(filtered_spectrogram, axis=1)

        # apply binary dilation to array with max values
        row_max = ndimage.morphology.binary_dilation(
            row_max, iterations=2).astype(row_max.dtype)

        rows_with_signal = row_max.sum()

        return rows_with_signal > signal_threshold, rows_with_signal < noise_threshold

    def save_spectrogram(self, target_file, spectrogram):
        # scale amplitude values to range [0, 255]
        # invert image so that black represents higher amplitudes
        img = self.scale_min_max(
            spectrogram, 255, 0, 0.0, 1.0).astype(numpy.uint8)

        # put low frequencies at the bottom of the image
        img = numpy.flip(img, axis=0)

        io.imsave(target_file, img)

        # copy spectrogram to cache
        cached_file_path = self.spectrogram_path.cached_file_path(
            "spectrograms", target_file)
        shutil.copy(target_file, cached_file_path)

    def get_cached_spectrograms(self, audio_file, include_noise_samples=True):
        spectrogram_name = f"{os.path.splitext(audio_file)[1]}_{self.chunk_length}"
        cache_path = self.spectrogram_path.cache("spectrograms")

        cached_spectrograms_for_current_file = []

        for cached_spectrogram in os.listdir(cache_path):
            if spectrogram_name in cached_spectrogram:
                if include_noise_samples or not "noise" in spectrogram_name:
                    spectrogram_path = os.path.join(
                        cache_path, cached_spectrogram)
                    cached_spectrograms_for_current_file.append(
                        spectrogram_path)

        return cached_spectrograms_for_current_file

    def create_spectrograms_from_file(self, audio_file, target_dir, include_noise_samples=True):
        cached_spectrograms_for_current_file = self.get_cached_spectrograms(
            audio_file, include_noise_samples)
        if len(cached_spectrograms_for_current_file) > 0:
            for file in cached_spectrograms_for_current_file:
                shutil.copy(file, target_dir)

        else:
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
                    mel_spectrogram_db)

                if contains_signal:
                    self.save_spectrogram(target_file, mel_spectrogram_db)
                elif is_noise and include_noise_samples:
                    target_file = os.path.join(
                        target_dir, "{}-{}_noise.png".format(file_name, i))
                    self.save_spectrogram(target_file, mel_spectrogram_db)

    def create_spectrograms_from_dir(self, audio_dir, target_dir, desc=None):
        # clean up target dir
        PathManager.empty_dir(target_dir)

        progress_bar = ProgressBar(
            os.listdir(audio_dir), desc="Create spectrograms for {}".format(desc), position=0,
            is_pipeline_run=self.spectrogram_path.is_pipeline_run)

        for file in progress_bar.iterable():
            if file.endswith(".mp3"):
                audio_path = os.path.join(audio_dir, file)
                self.create_spectrograms_from_file(audio_path, target_dir)

    def create_spectrograms_for_splits(self, splits=None, clear_spectrogram_cache=False):
        if splits is None:
            splits = ["train", "val", "test"]

        descriptions = {
            "train": "training set",
            "val": "validation set",
            "test": "test set"
        }

        if clear_spectrogram_cache:
            self.spectrogram_path.clear_cache("spectrograms")

        for split in splits:
            spectrogram_dir = self.spectrogram_path.data_folder(
                split, "spectrograms", chunk_length=self.chunk_length)
            audio_dir = self.audio_path.data_folder(split, "audio")
            audio_label_file = self.audio_path.audio_label_file(split)
            PathManager.ensure_dir(spectrogram_dir)
            self.create_spectrograms_from_dir(
                audio_dir, spectrogram_dir, descriptions[split])
            self.create_spectrogram_labels(audio_label_file, spectrogram_dir)

    def create_spectrogram_labels(self, label_file, spectrogram_dir):
        labels = pd.read_json(label_file)

        spectrogram_labels = []

        for file in os.listdir(spectrogram_dir):
            if file.endswith(".png"):
                file_id = int(file.split("-")[0])
                label = labels[labels["id"] == file_id]

                if file.endswith("noise.png"):
                    label["label"] = "noise"
                    label["sound_type"] = "noise"

                if len(label) != 1:
                    raise NameError(
                        "No matching labels found for file with id {}".format(file_id))

                label["file_name"] = file

                spectrogram_labels.append(label)

        if len(spectrogram_labels) > 0:
            spectrogram_labels = pd.concat(spectrogram_labels).sort_values(
                by=['file_name'])

            spectrogram_labels.to_json(label_file.replace(
                ".json", "_{}.json".format(self.chunk_length)), "records", indent=4)
        else:
            raise NameError("No spectrograms found")
