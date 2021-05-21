import librosa
import math
import numpy
import os
from skimage import io
import tqdm
import warnings

warnings.filterwarnings('ignore')


class SpectrogramCreator():
    def __init__(self, chunk_length, path_manager):
        # parameters for spectorgram creation
        self.chunk_length = chunk_length  # chunk length in seconds
        self.sampling_rate = 44100  # number of samples per second
        self.window_length = 1024
        self.n_fft = 1024
        self.hop_length = 196
        self.fmin = 500  # minimum frequency
        self.fmax = 15000  # maximum frequency

        self.samples_per_chunk = self.sampling_rate * chunk_length

        self.path = path_manager

    def scale_min_max(self, x, min=0.0, max=1.0):
        x_std = (x - x.min()) / (x.max() - x.min())
        x_scaled = x_std * (max - min) + min
        return x_scaled

    def save_spectrogram(self, target_file, spectrogram):
        # scale amplitude values to range [0, 1]
        img = self.scale_min_max(spectrogram, 0, 255).astype(numpy.uint8)

        # put low frequencies at the bottom of the image
        img = numpy.flip(img, axis=0)

        # invert image so that black represents higher amplitudes
        img = 255-img

        io.imsave(target_file, img)

    def create_spectrograms_from_file(self, audio_file, target_dir):

        # load audio file
        amplitudes, sr = librosa.load(audio_file, sr=self.sampling_rate)

        audio_length = amplitudes.shape[0]

        number_of_chunks = math.floor(
            audio_length / self.samples_per_chunk)

        # split audio file in chunks and create one spectrogram per chunk
        for i in range(number_of_chunks):
            # get samples of current chunk
            chunk = amplitudes[i *
                               self.samples_per_chunk:(i+1)*self.samples_per_chunk]

            # apply short time fourier transformation to extract frequency information from amplitude data
            mel_spectrogram = librosa.feature.melspectrogram(chunk, sr=self.sampling_rate, hop_length=self.hop_length, n_fft=self.n_fft,
                                                             win_length=self.window_length, n_mels=112, fmin=self.fmin, fmax=self.fmax)

            # convert power spectrogram to dB-scaled spectrogram
            mel_spectrogram_dB = librosa.power_to_db(
                mel_spectrogram, ref=numpy.max)

            file_name = os.path.splitext(os.path.basename(audio_file))[0]
            target_file = os.path.join(
                target_dir, "{}-{}.png".format(file_name, i))

            self.save_spectrogram(target_file, mel_spectrogram_dB)

    def create_spectrograms_from_dir(self, audio_dir, target_dir, desc=None):
        progress_bar = tqdm.tqdm(
            total=len(os.listdir(audio_dir)), desc="Create spectrograms for {}".format(desc), position=0)

        for file in os.listdir(audio_dir):
            if file.endswith(".mp3"):
                audio_path = os.path.join(audio_dir, file)
                self.create_spectrograms_from_file(audio_path, target_dir)
            progress_bar.update(1)

    def create_spectrograms_for_datasets(self, datasets=["train", "val", "test"]):
        dirs = []

        if "train" in datasets:
            train_spectrogram_dir = os.path.join(
                self.path.train_dir,  "spectrograms_{}".format(self.chunk_length * 1000))
            dirs.append([self.path.train_audio_dir,
                        train_spectrogram_dir, "training set"])

        if "val" in datasets:
            val_spectrogram_dir = os.path.join(
                self.path.val_dir,  "spectrograms_{}".format(self.chunk_length * 1000))
            dirs.append([self.path.val_audio_dir,
                        val_spectrogram_dir, "validation set"])

        if "test" in datasets:
            test_spectrogram_dir = os.path.join(
                self.path.test_dir,  "spectrograms_{}".format(self.chunk_length * 1000))
            dirs.append([self.path.test_audio_dir,
                        test_spectrogram_dir, "test set"])

        for audio_dir, spectrogram_dir, desc in dirs:
            self.path.ensure_dir(spectrogram_dir)
            self.create_spectrograms_from_dir(
                audio_dir, spectrogram_dir, desc)
