from kubeflow import fairing
import logging
import os
import subprocess
import sys

from kubeflow_utils.config import settings


class PathManager:
    def __init__(self, data_dir: str, gcs_path=None):
        self.data_dir = data_dir
        self.cache_dir = os.path.join(self.data_dir, "cache")

        self.cache_dirs = {
            "audio": os.path.join(self.cache_dir, "audio"),
            "labels": os.path.join(self.cache_dir, "labels"),
            "spectrograms": os.path.join(self.cache_dir, "spectrograms"),
        }

        self.data_dirs = {
            "train": os.path.join(self.data_dir, "train"),
            "val": os.path.join(self.data_dir, "val"),
            "test": os.path.join(self.data_dir, "test"),
        }

        self.data_subdirs = ["audio"]

        self.ensure_dirs(self.cache_dirs.values())
        self.ensure_dirs(self.data_dirs.values())

        for dir in self.data_dirs.values():
            for subdir in self.data_subdirs:
                self.ensure_dir(os.path.join(dir, subdir))

        # google cloud storage config
        if gcs_path:
            self.use_gcs = True
            self.GCP_PROJECT = fairing.cloud.gcp.guess_project_name()
            self.GCS_BUCKET_ID = f'{settings.gcloud.bucket_id}'
            self.GCS_BUCKET = f'{settings.gcloud.bucket_prefix}{self.GCS_BUCKET_ID}'
            self.GCS_BUCKET_PATH = f'{self.GCS_BUCKET}/{settings.gcloud.bucket_path}'

            if not self.gcs_bucket_exists(self.GCS_BUCKET_ID):
                self.gcs_make_bucket(self.GCS_BUCKET, self.GCP_PROJECT)

            self.gcs_cache_dir = os.path.join(self.GCS_BUCKET, "cache")
            self.gcs_dirs = {
                "audio_cache": os.path.join(self.gcs_cache_dir, "audio"),
                "labels_cache": os.path.join(self.gcs_cache_dir, "labels"),
                "spectrograms_cache": os.path.join(self.gcs_cache_dir, "spectrograms"),
            }

    def ensure_dir(self, dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def ensure_dirs(self, dir_paths):
        for dir_path in dir_paths:
            self.ensure_dir(dir_path)

    def empty_dir(self, dir_path):
        for file in os.listdir(dir_path):
            os.remove(os.path.join(dir_path, file))

    def audio_label_file(self, split: str):
        return os.path.join(self.data_dirs[split], f"{split}.json")

    def spectrogram_label_file(self, split: str, **kwargs):
        keywords = ""
        for key in kwargs.values():
            keywords += f"_{key}"
        return os.path.join(self.data_dirs[split], f"{split}{keywords}.json")

    def cache(self, subdir: str):
        return self.cache_dirs[subdir]

    def data_folder(self, split: str, subdir: str, **kwargs):
        if subdir == "spectrograms":
            for key in kwargs.values():
                subdir += f"_{key}"
        return os.path.join(self.data_dirs[split], subdir)

    def gcs_copy_file(self, src_path: str, dest_path: str):
        if sys.platform == 'win32' or sys.platform == 'nt':
            logging.info(
                subprocess.run(['gsutil', 'cp', src_path, dest_path], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8'))
        else:
            logging.info(
                subprocess.run(['gsutil', 'cp', src_path, dest_path], stdout=subprocess.PIPE, shell=True).stdout[:-1].decode(
                    'utf-8'))
        logging.info(f'Copied {src_path} to {dest_path}')

    def gcs_copy_dir(self, src_path: str, dest_path: str):
        if sys.platform == 'win32' or sys.platform == 'nt':
            logging.info(
                subprocess.run(['gsutil', '-m', 'cp', '-r', src_path, dest_path], stdout=subprocess.PIPE, shell=True).stdout[
                    :-1].decode('utf-8'))
        else:
            logging.info(
                subprocess.run(['gsutil', '-m', 'cp', '-r', src_path, dest_path], stdout=subprocess.PIPE).stdout[:-1].decode(
                    'utf-8'))
        logging.info(f'Copied {src_path} to {dest_path}')

    def gcs_bucket_path(self, bucket_name: str):
        return f'{settings.gcloud.bucket_prefix}{bucket_name}/'

    def gcs_make_bucket(self, bucket_name: str, project_name: str):
        if sys.platform == 'win32' or sys.platform == 'nt':
            logging.info(
                subprocess.run(['gsutil', 'mb', '-p', project_name, bucket_name], stdout=subprocess.PIPE,
                               shell=True).stdout[:-1].decode('utf-8'))
        else:
            logging.info(subprocess.run(['gsutil', 'mb', '-p', project_name, bucket_name], stdout=subprocess.PIPE).stdout[
                :-1].decode('utf-8'))
        logging.info(f'Created bucket {bucket_name}')

    def gcs_bucket_exists(self, bucket_name: str):
        bucket_path = self.gcs_bucket_path(bucket_name)
        if sys.platform == 'win32' or sys.platform == 'nt':
            result = subprocess.run(['gsutil', 'ls', '-b', bucket_path],
                                    stdout=subprocess.PIPE, shell=True).stdout[:-1].decode('utf-8')
        else:
            result = subprocess.run(
                ['gsutil', 'ls', '-b', bucket_path], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8')
        return result == bucket_path

    def copy_cache_to_gcs(self, subdir: str):
        self.gcs_copy_dir(self.cache(subdir), self.GCS_BUCKET)

    def copy_cache_from_gcs(self, subdir: str):
        self.gcs_copy_dir(self.gcs_dirs[f"{subdir}_cache"], self.data_dir)
