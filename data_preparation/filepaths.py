import os


class PathManager():
    def __init__(self, data_dir):
        self.cache_dir = os.path.join(data_dir, "cache")
        self.audio_cache_dir = os.path.join(self.cache_dir, "audio")
        self.label_cache_dir = os.path.join(self.cache_dir, "labels")

        self.train_dir = os.path.join(data_dir, "train")
        self.train_audio_dir = os.path.join(self.train_dir, "audio")
        self.test_dir = os.path.join(data_dir, "test")
        self.test_audio_dir = os.path.join(self.test_dir, "audio")
        self.val_dir = os.path.join(data_dir, "val")
        self.val_audio_dir = os.path.join(self.val_dir, "audio")

        # create outer directories
        self.ensure_dirs([self.cache_dir, self.train_dir,
                         self.test_dir, self.val_dir])

        # create nested directories
        self.ensure_dirs([self.audio_cache_dir, self.label_cache_dir, self.train_audio_dir, self.test_audio_dir,
                          self.val_audio_dir])

    def ensure_dir(self, dir_path):
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

    def ensure_dirs(self, dir_paths):
        for dir_path in dir_paths:
            self.ensure_dir(dir_path)

    def empty_dir(self, dir_path):
        for file in os.listdir(dir_path):
            os.remove(os.path.join(dir_path, file))

    def train_spectrogram_dir(self, chunk_length):
        return os.path.join(self.train_dir, "spectrograms_{}".format(chunk_length))

    def val_spectrogram_dir(self, chunk_length):
        return os.path.join(self.val_dir, "spectrograms_{}".format(chunk_length))

    def test_spectrogram_dir(self, chunk_length):
        return os.path.join(self.test_dir, "spectrograms_{}".format(chunk_length))
