import os
import subprocess
import shutil
from typing import List

from general.logging import logger
from kubeflow import fairing


class PathManager:
    @staticmethod
    def copytree(src, dest, symlinks=False, ignore=None):
        # see https://stackoverflow.com/a/12514470
        for item in os.listdir(src):
            source_file = os.path.join(src, item)
            dest_file = os.path.join(dest, item)
            if os.path.isdir(source_file):
                shutil.copytree(source_file, dest_file, symlinks, ignore)
            else:
                shutil.copy2(source_file, dest_file)

    @staticmethod
    def ensure_dir(dir_path: str):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    @staticmethod
    def ensure_dirs(dir_paths: [str]):
        for dir_path in dir_paths:
            PathManager.ensure_dir(dir_path)

    @staticmethod
    def empty_dir(dir_path: str):
        logger.verbose("empty dir %s", dir_path)
        # from https://gist.github.com/jagt/6759127
        for root, dirs, files in os.walk(dir_path, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                os.remove(file_path)
            for name in dirs:
                subdir_path = os.path.join(root, name)
                os.rmdir(subdir_path)

    @staticmethod
    def ensure_trailing_slash(path: str):
        if not path.endswith("/"):
            return path + "/"
        return path

    @staticmethod
    def gcs_copy_file(src_path: str, dest_path: str, quiet: bool = True):
        try:
            subprocess.run(["gsutil", "-q", "cp", src_path, dest_path], check=True)
            if not quiet:
                logger.info(f"Copied {src_path} to {dest_path}")
        except subprocess.CalledProcessError:
            error_message = f"Copying {src_path} to {dest_path} failed"
            logger.error(error_message)
            raise NameError(error_message)

    @staticmethod
    def gcs_copy_files(src_paths: List[str], dest_path: str, quiet: bool = True):
        if len(src_paths) > 0:
            try:
                subprocess.run(["gsutil", "-m", "-q", "cp", "-I", dest_path], check=True,
                               input="\n".join(src_paths), text=True)
                if not quiet:
                    logger.info(f"Copied files to {dest_path}")
            except subprocess.CalledProcessError:
                error_message = f"Copying files to {dest_path} failed"
                logger.error(error_message)
                raise NameError(error_message)

    @staticmethod
    def gcs_remove_dir(dir_path: str):
        if PathManager.gcs_file_exists(dir_path):
            try:
                subprocess.run(["gsutil", "-q", "rm", "-r", dir_path], check=True)
                logger.info(f"Removed {dir_path}")
            except subprocess.CalledProcessError:
                error_message = f"Removing dir {dir_path} failed"
                logger.error(error_message)
                raise NameError(error_message)

    @staticmethod
    def gcs_copy_dir(src_path: str, dest_path: str):
        src_path = PathManager.ensure_trailing_slash(src_path)
        try:
            if src_path.startswith("gs://"):
                subprocess.run(["gsutil", "-m", "rsync", "-r", src_path, dest_path], check=True)
            else:
                subprocess.run(["gsutil", "-q", "-m", "cp", "-n", "-r", f"{src_path}*", dest_path], check=True)
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
            # if the bucket does not exist, this will throw a BucketNotFoundException
            subprocess.run(
                ["gsutil", "ls", "-b", bucket_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            logger.info(f"Bucket {bucket_path} exists")
            return True
        except subprocess.CalledProcessError:
            logger.info(f"Bucket {bucket_path} does not exist")
            return False

    @staticmethod
    def gcs_file_exists(file_path: str, quiet: bool = True):
        try:
            result = subprocess.run(
                ["gsutil", "ls", file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if not quiet:
                logger.info(f"File {file_path} exists")
            return True
        except subprocess.CalledProcessError:
            if not quiet:
                logger.info(f"File {file_path} does not exist")
            return False

    def __init__(self, data_dir: str, gcs_bucket: str = None):
        self.data_dir = data_dir
        self.cache_dir = os.path.join(self.data_dir, "cache")
        self.is_pipeline_run = False

        # google cloud storage config
        if gcs_bucket:
            self.is_pipeline_run = True
            self.GCP_PROJECT = fairing.cloud.gcp.guess_project_name()
            self.GCS_BUCKET_ID = gcs_bucket
            self.GCS_BUCKET = f"gs://{self.GCS_BUCKET_ID}"

            if not PathManager.gcs_bucket_exists(self.GCS_BUCKET):
                PathManager.gcs_make_bucket(self.GCS_BUCKET, self.GCP_PROJECT)

            self.gcs_cache_dir = os.path.join(self.GCS_BUCKET, "cache")
            self.gcs_models_dir = os.path.join(self.GCS_BUCKET, "models")

    def label_file(self, split: str, **kwargs):
        self.ensure_dir(os.path.join(self.data_dir, split))
        keywords = ""
        for key in kwargs.values():
            keywords += f"_{key}"
        return os.path.join(self.data_dir, split, f"{split}{keywords}.csv")

    def categories_file(self):
        return os.path.join(self.data_dir, "categories.txt")

    def cache(self, subdir: str, **kwargs):
        for key in kwargs.values():
            subdir += f"_{key}"
        cache_dir = os.path.join(self.cache_dir, subdir)
        PathManager.ensure_dir(cache_dir)
        return cache_dir

    def gcs_cache(self, subdir: str, **kwargs):
        for key in kwargs.values():
            subdir += f"_{key}"
        return PathManager.ensure_trailing_slash(os.path.join(self.gcs_cache_dir, subdir))

    def cached_file_path(self, subdir: str, file_path: str, **kwargs):
        file_name = os.path.basename(file_path)
        return os.path.join(self.cache(subdir, **kwargs), file_name)

    def data_folder(self, split: str, subdir: str = "", **kwargs):
        if subdir == "spectrograms":
            for key in kwargs.values():
                subdir += f"_{key}"
        data_folder_path = os.path.join(self.data_dir, split, subdir)
        PathManager.ensure_dir(data_folder_path)
        return data_folder_path

    def model_dir(self):
        return os.path.join(self.data_dir, "models")

    def gcs_model_dir(self):
        return self.gcs_models_dir

    def copy_cache_to_gcs(self, subdir: str, **kwargs):
        PathManager.gcs_copy_dir(self.cache(subdir), self.gcs_cache(subdir, **kwargs))

    def copy_cache_from_gcs(self, subdir: str, **kwargs):
        if self.gcs_file_exists(self.gcs_cache(subdir, **kwargs)):
            PathManager.gcs_copy_dir(self.gcs_cache(subdir, **kwargs), self.cache(subdir, **kwargs))

    def copy_file_to_gcs_cache(self, file_path: str, subdir: str, **kwargs):
        if type(file_path) == list:
            PathManager.gcs_copy_files(file_path, self.gcs_cache(subdir, **kwargs))
        else:
            PathManager.gcs_copy_file(file_path, self.gcs_cache(subdir, **kwargs))

    def clear_cache(self, subdir: str, **kwargs):
        PathManager.empty_dir(self.cache(subdir, **kwargs))
        if self.is_pipeline_run:
            PathManager.gcs_remove_dir(self.gcs_cache(subdir, **kwargs))
