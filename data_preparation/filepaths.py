from kubeflow import fairing
import logging
import os
import subprocess

from kubeflow_utils.config import settings


class PathManager:
    def __init__(self, data_dir, gcs_path=None):
        self.ensure_dir(data_dir)

        self.cache_dir = os.path.join(data_dir, "cache")
        self.audio_cache_dir = os.path.join(self.cache_dir, "audio")
        self.label_cache_dir = os.path.join(self.cache_dir, "labels")

        self.train_dir = os.path.join(data_dir, "train")
        self.train_audio_dir = os.path.join(self.train_dir, "audio")
        self.test_dir = os.path.join(data_dir, "test")
        self.test_audio_dir = os.path.join(self.test_dir, "audio")
        self.val_dir = os.path.join(data_dir, "val")
        self.val_audio_dir = os.path.join(self.val_dir, "audio")

        # create nested directories
        self.ensure_dirs([self.audio_cache_dir, self.label_cache_dir, self.train_audio_dir, self.test_audio_dir,
                          self.val_audio_dir])

        # google cloud storage config
        if gcs_path:
            self.use_gcs = True
        self.GCP_PROJECT = fairing.cloud.gcp.guess_project_name()
        self.GCS_BUCKET_ID = f'{settings.gcloud.bucket_id}'
        self.GCS_BUCKET = f'{settings.gcloud.bucket_prefix}{self.GCS_BUCKET_ID}'
        self.GCS_BUCKET_PATH = f'{self.GCS_BUCKET}/{settings.gcloud.bucket_path}'

        self.gcs_cache_dir = os.path.join(self.GCS_BUCKET_PATH, "cache")
        self.gcs_audio_cache_dir = os.path.join(self.cache_dir, "audio")
        self.gcs_label_cache_dir = os.path.join(self.cache_dir, "labels")

        self.gcs_train_dir = os.path.join(self.GCS_BUCKET_PATH, "train")
        self.gcs_train_audio_dir = os.path.join(self.train_dir, "audio")
        self.gcs_test_dir = os.path.join(self.GCS_BUCKET_PATH, "test")
        self.gcs_test_audio_dir = os.path.join(self.test_dir, "audio")
        self.gcs_val_dir = os.path.join(self.GCS_BUCKET_PATH, "val")
        self.gcs_val_audio_dir = os.path.join(self.val_dir, "audio")

    def ensure_dir(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def ensure_dirs(self, dir_paths):
        for dir_path in dir_paths:
            self.ensure_dir(dir_path)

    def empty_dir(self, dir_path):
        for file in os.listdir(dir_path):
            os.remove(os.path.join(dir_path, file))

    def train_label_file(self):
        return os.path.join(self.train_dir, "train.json")

    def val_label_file(self):
        return os.path.join(self.val_dir, "val.json")

    def test_label_file(self):
        return os.path.join(self.test_dir, "test.json")

    def train_spectrogram_dir(self, chunk_length):
        return os.path.join(self.train_dir, "spectrograms_{}".format(chunk_length))

    def val_spectrogram_dir(self, chunk_length):
        return os.path.join(self.val_dir, "spectrograms_{}".format(chunk_length))

    def test_spectrogram_dir(self, chunk_length):
        return os.path.join(self.test_dir, "spectrograms_{}".format(chunk_length))

    def gcs_copy_file(self, src_path, dest_path):
        logging.info(
            subprocess.run(['gsutil', 'cp', src_path, dest_path], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8'))
        logging.info(f'Copied {src_path} to {dest_path}')

    def gcs_copy_dir(self, src_path, dest_path):
        logging.info(
            subprocess.run(['gsutil', 'cp', '-r', src_path, dest_path], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8'))
        logging.info(f'Copied {src_path} to {dest_path}')

    def copy_cache_to_gcs(self):
        self.gcs_copy_dir(self.cache_dir, self.gcs_cache_dir)

    def copy_cache_from_gcs(self):
        self.gcs_copy_dir(self.gcs_cache_dir, self.cache_dir)
