from kubeflow import fairing
import os
import subprocess
import sys

from kubeflow_utils.config import settings
from general.logging import logger


class PathManager:
    @staticmethod
    def ensure_dir(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    @staticmethod
    def ensure_dirs(dir_paths):
        for dir_path in dir_paths:
            PathManager.ensure_dir(dir_path)

    @staticmethod
    def empty_dir(dir_path):
        for file in os.listdir(dir_path):
            os.remove(os.path.join(dir_path, file))

    @staticmethod
    def ensure_trailing_slash(path: str):
        if not path.endswith("/"):
            return path + "/"
        return path

    @staticmethod
    def gcs_copy_file(src_path: str, dest_path: str):
        try:
            subprocess.run(["gsutil", "-q", "cp", src_path, dest_path], check=True)
            logger.info(f"Copied {src_path} to {dest_path}")
        except subprocess.CalledProcessError:
            error_message = f"Copying {src_path} to {dest_path} failed"
            logger.error(error_message)
            raise NameError(error_message)

    @staticmethod
    def gcs_copy_dir(src_path: str, dest_path: str):
        src_path = PathManager.ensure_trailing_slash(src_path)
        try:
            subprocess.run(["gsutil", "-q", "-m", "cp", "-r", f"{src_path}*", dest_path], check=True)
            logger.info(f"Copied {src_path} to {dest_path}")
        except subprocess.CalledProcessError:
            error_message = f"Copying {src_path} to {dest_path} failed"
            logger.error(error_message)
            raise NameError(error_message)

    @staticmethod
    def gcs_make_bucket(bucket_path: str, project_name: str):
        try:
            subprocess.run(["gsutil", "-q", "mb", "-p", project_name, bucket_path], check=True)
            logger.info(f"Created bucket {bucket_path}")
        except subprocess.CalledProcessError:
            error_message = f"Creating bucket {bucket_path} failed"
            logger.error(error_message)
            raise NameError(error_message)

    @staticmethod
    def gcs_bucket_exists(bucket_path: str):
        bucket_path = PathManager.ensure_trailing_slash(bucket_path)
        try:
            if sys.platform == "win32" or sys.platform == "nt":
                result = subprocess.run(["gsutil", "ls", "-b", bucket_path],
                                        stdout=subprocess.PIPE, shell=True, check=True).stdout[:-1].decode("utf-8")
            else:
                result = subprocess.run(
                    ["gsutil", "ls", "-b", bucket_path], stdout=subprocess.PIPE, check=True).stdout[:-1].decode("utf-8")
            bucket_exists = (result == bucket_path)
            if bucket_exists:
                logger.info(f"Bucket {bucket_path} exists")
            else:
                logger.info(f"Bucket {bucket_path} does not exist")
            return bucket_exists
        except subprocess.CalledProcessError:
            error_message = f"Failed to check existence of bucket {bucket_path}"
            logger.error(error_message)
            raise NameError(error_message)

    @staticmethod
    def gcs_file_exists(file_path: str):
        try:
            if sys.platform == "win32" or sys.platform == "nt":
                result = subprocess.run(["gsutil", "ls", file_path],
                                        stdout=subprocess.PIPE, shell=True, check=True).stdout[:-1].decode("utf-8")
            else:
                result = subprocess.run(
                    ["gsutil", "ls", file_path], stdout=subprocess.PIPE, check=True).stdout[:-1].decode("utf-8")
            file_exists = file_path in result
            if file_exists:
                logger.info(f"File {file_path} exists")
            else:
                logger.info(f"File {file_path} does not exist")
            return file_exists
        except subprocess.CalledProcessError:
            error_message = f"Failed to check existence of file {file_path}"
            logger.error(error_message)
            raise NameError(error_message)

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

        for dir_path in self.data_dirs.values():
            for subdir in self.data_subdirs:
                self.ensure_dir(os.path.join(dir_path, subdir))

        self.is_pipeline_run = False

        # google cloud storage config
        if gcs_path:
            self.is_pipeline_run = True
            self.GCP_PROJECT = fairing.cloud.gcp.guess_project_name()
            self.GCS_BUCKET_ID = f"{settings.gcloud.bucket_id}"
            self.GCS_BUCKET = f"{settings.gcloud.bucket_prefix}{self.GCS_BUCKET_ID}"
            self.GCS_BUCKET_PATH = f"{self.GCS_BUCKET}/{settings.gcloud.bucket_path}"

            if not PathManager.gcs_bucket_exists(self.GCS_BUCKET):
                PathManager.gcs_make_bucket(self.GCS_BUCKET, self.GCP_PROJECT)

            self.gcs_cache_dir = os.path.join(self.GCS_BUCKET, "cache")
            self.gcs_dirs = {
                "audio_cache": os.path.join(self.gcs_cache_dir, "audio", ""),
                "labels_cache": os.path.join(self.gcs_cache_dir, "labels", ""),
                "spectrograms_cache": os.path.join(self.gcs_cache_dir, "spectrograms", ""),
            }

    def audio_label_file(self, split: str):
        return os.path.join(self.data_dirs[split], f"{split}.json")

    def spectrogram_label_file(self, split: str, **kwargs):
        keywords = ""
        for key in kwargs.values():
            keywords += f"_{key}"
        return os.path.join(self.data_dirs[split], f"{split}{keywords}.json")

    def cache(self, subdir: str):
        return self.cache_dirs[subdir]

    def cached_file_path(self, subdir, file_path):
        file_name = os.path.basename(file_path)
        return os.path.join(self.cache(subdir), file_name)

    def data_folder(self, split: str, subdir: str, **kwargs):
        if subdir == "spectrograms":
            for key in kwargs.values():
                subdir += f"_{key}"
        return os.path.join(self.data_dirs[split], subdir)

    def copy_cache_to_gcs(self, subdir: str):
        PathManager.gcs_copy_dir(self.cache(subdir), self.gcs_dirs[f"{subdir}_cache"])

    def copy_cache_from_gcs(self, subdir: str):
        if self.gcs_file_exists(self.gcs_dirs[f"{subdir}_cache"]):
            PathManager.gcs_copy_dir(self.gcs_dirs[f"{subdir}_cache"], self.data_dir)
