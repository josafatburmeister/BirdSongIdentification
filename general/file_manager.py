import os
import shutil
import subprocess
from typing import List, Union

from general.logging import logger
from kubeflow import fairing


class FileManager:
    """
    Manages files within a given directory. As described in the Readme, our pipeline uses specific directory structures
    to transfer data between pipeline stages. This class implements the file operations and file path handling necessary
    to create these directory structures.
    """

    @staticmethod
    def copytree(src: str, dest: str, symlinks: bool = False, ignore: bool = None) -> None:
        """
        Recursively copies a directory and its subdirectories.
    
        Args:
            src: Path of the source directory.
            dest: Path of the destination directory.
            symlinks: See shutil documentation.
            ignore: See shutil documentation.

        Returns:
            None
        """

        # see https://stackoverflow.com/a/12514470
        for file_name in os.listdir(src):
            source_file = os.path.join(src, file_name)
            dest_file = os.path.join(dest, file_name)
            if os.path.isdir(source_file):
                shutil.copytree(source_file, dest_file, symlinks, ignore)
            else:
                shutil.copy2(source_file, dest_file)

    @staticmethod
    def empty_dir(dir_path: str) -> None:
        """
        Deletes a directory including all subdirectories.
    
        Args:
            dir_path: Path of the directory to be deleted.

        Returns:
            None
        """

        logger.verbose("Empty dir %s.", dir_path)
        # from https://gist.github.com/jagt/6759127
        for root, dirs, files in os.walk(dir_path, topdown=False):
            for name in files:
                file_path = os.path.join(root, name)
                os.remove(file_path)
            for name in dirs:
                subdir_path = os.path.join(root, name)
                os.rmdir(subdir_path)

    @staticmethod
    def ensure_trailing_slash(path: str) -> str:
        """
        Ensures that a file path ends with a trailing slash.

        Args:
            path: File path.

        Returns:
            File path with trailing slash.
        """

        if not path.endswith("/"):
            return path + "/"
        return path

    @staticmethod
    def gcs_copy_file(src_path: str, dest_path: str, quiet: bool = True) -> None:
        """
        Copies a file from or to a GCS bucket.

        Args:
            src_path: Path of the source file (for paths in GCS buckets full paths in format
                gs://<bucket-name>/<file-name> are expected).
            dest_path: Path of the destination file (for paths in GCS buckets full paths in format
                gs://<bucket-name>/<file-name> are expected).
            quiet: Whether success or error messages should be logged.

        Returns:
            None
        """

        try:
            subprocess.run(["gsutil", "-q", "cp", src_path, dest_path], check=True)
            if not quiet:
                logger.info(f"Copied {src_path} to {dest_path}.")
        except subprocess.CalledProcessError:
            error_message = f"Copying {src_path} to {dest_path} failed."
            logger.error(error_message)
            raise NameError(error_message)

    @staticmethod
    def gcs_copy_files(src_paths: List[str], dest_path: str, quiet: bool = True) -> None:
        """
        Copies multiple files to or from a GCS bucket.

        Args:
            src_paths: List with paths of the files to be copied (for paths in GCS buckets full paths in format
                gs://<bucket-name>/<file-name> are expected).
            dest_path: Path of destination directory (for paths in GCS buckets full paths in format
                gs://<bucket-name>/<file-name> are expected).
            quiet: Whether success or error messages should be logged.

        Returns:
            None
        """

        if len(src_paths) > 0:
            try:
                subprocess.run(["gsutil", "-m", "-q", "cp", "-I", dest_path], check=True,
                               input="\n".join(src_paths), text=True)
                if not quiet:
                    logger.info(f"Copied files to {dest_path}.")
            except subprocess.CalledProcessError:
                error_message = f"Copying files to {dest_path} failed."
                logger.error(error_message)
                raise NameError(error_message)

    @staticmethod
    def gcs_remove_dir(dir_path: str) -> None:
        """
        Deletes a directory from a GCS bucket.

        Args:
            dir_path: Path of the directory to be deleted.

        Returns:
            None
        """

        if FileManager.gcs_file_exists(dir_path):
            try:
                subprocess.run(["gsutil", "-q", "rm", "-r", dir_path], check=True)
                logger.info(f"Removed {dir_path}.")
            except subprocess.CalledProcessError:
                error_message = f"Removing dir {dir_path} failed."
                logger.error(error_message)
                raise NameError(error_message)

    @staticmethod
    def gcs_copy_dir(src_path: str, dest_path: str) -> None:
        """
        Copies a directory from or to a GCS bucket.

        Args:
            src_path: Path of the source directory (for paths in GCS buckets full paths in format
                gs://<bucket-name>/<file-name> are expected).
            dest_path: Path of the destination directory (for paths in GCS buckets full paths in format
                gs://<bucket-name>/<file-name> are expected).

        Returns:
            None
        """
        src_path = FileManager.ensure_trailing_slash(src_path)
        try:
            if src_path.startswith("gs://"):
                subprocess.run(["gsutil", "-m", "rsync", "-r", src_path, dest_path], check=True)
            else:
                subprocess.run(["gsutil", "-q", "-m", "cp", "-n", "-r", f"{src_path}*", dest_path], check=True)
            logger.info(f"Copied {src_path} to {dest_path}.")
        except subprocess.CalledProcessError:
            error_message = f"Copying {src_path} to {dest_path} failed."
            logger.error(error_message)
            raise NameError(error_message)

    @staticmethod
    def gcs_make_bucket(bucket_path: str, project_name: str) -> None:
        """
        Creates a GCS bucket.

        Args:
            bucket_path: Path of the GCS bucket to be created (a full path in format gs://<bucket-name> is expected).
            project_name: Name of the project in which the bucket is to be created.

        Returns:
            None
        """

        try:
            subprocess.run(["gsutil", "-q", "mb", "-p", project_name, bucket_path], check=True)
            logger.info(f"Created bucket {bucket_path}.")
        except subprocess.CalledProcessError:
            error_message = f"Creating bucket {bucket_path} failed."
            logger.error(error_message)
            raise NameError(error_message)

    @staticmethod
    def gcs_bucket_exists(bucket_path: str) -> bool:
        """
        Determines whether the given GCS bucket exists.

        Args:
            bucket_path: Path of the GCS bucket (a full path in format gs://<bucket-name> is expected).

        Returns:
            True if the bucket exists, otherwise False.
        """

        bucket_path = FileManager.ensure_trailing_slash(bucket_path)
        try:
            # if the bucket does not exist, this will throw a BucketNotFoundException
            subprocess.run(
                ["gsutil", "ls", "-b", bucket_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            logger.info(f"Bucket {bucket_path} exists.")
            return True
        except subprocess.CalledProcessError:
            logger.info(f"Bucket {bucket_path} does not exist.")
            return False

    @staticmethod
    def gcs_file_exists(file_path: str, quiet: bool = True) -> bool:
        """
        Determines whether a file exists within a GCS bucket.

        Args:
            file_path: File path (a full path in format gs://<bucket-name>/<file-name> is expected).
            quiet: Whether success or error messages should be logged.

        Returns:
            True if the bucket exists, otherwise False.
        """

        try:
            subprocess.run(
                ["gsutil", "ls", file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            if not quiet:
                logger.info(f"File {file_path} exists.")
            return True
        except subprocess.CalledProcessError:
            if not quiet:
                logger.info(f"File {file_path} does not exist.")
            return False

    def __init__(self, base_dir: str, gcs_bucket: str = None) -> None:
        """

        Args:
            base_dir: Path of the directory where the directory structure described in the Readme is to be created.
            gcs_bucket: Name of a GCS bucket to be used for persisting files (e.g. the cache and model files).
        """
        self.base_dir = base_dir
        self.cache_dir = os.path.join(self.base_dir, "cache")
        self.is_pipeline_run = False

        # google cloud storage config
        if gcs_bucket:
            self.is_pipeline_run = True
            self.GCP_PROJECT = fairing.cloud.gcp.guess_project_name()
            self.GCS_BUCKET_ID = gcs_bucket
            self.GCS_BUCKET = f"gs://{self.GCS_BUCKET_ID}"

            if not FileManager.gcs_bucket_exists(self.GCS_BUCKET):
                FileManager.gcs_make_bucket(self.GCS_BUCKET, self.GCP_PROJECT)

            self.gcs_cache_dir = os.path.join(self.GCS_BUCKET, "cache")
            self.gcs_model_dir = os.path.join(self.GCS_BUCKET, "models")

    def label_file(self, dataset: str, label_type: str) -> str:
        """
        Computes the absolute path of the label file for the given dataset. Creates all subdirectories of this path that
        do not exist yet.

        Args:
            dataset: Name of a dataset (e.g. "train", "val", "test").
            label_type: Type of label file (e.g. "audio", "spectrograms").

        Returns:
            Absolute path of the label file for the given dataset.
        """

        os.makedirs(os.path.join(self.base_dir, dataset), exist_ok=True)
        return os.path.join(self.base_dir, dataset, f"{label_type}.csv")

    def categories_file(self) -> str:
        """

        Returns:
            Absolute path of the categories file.
        """

        return os.path.join(self.base_dir, "categories.txt")

    def cache(self, subdir: str, **kwargs) -> str:
        """
        Computes the absolute path of a subdirectory of the cache directory. Creates all subdirectories of this path
        that do not exist yet.

        Args:
            subdir: Name of the cache subdirectory.

        Returns:
            Absolute path of the cache subdirectory.
        """

        for key in kwargs.values():
            subdir += f"_{key}"
        cache_dir = os.path.join(self.cache_dir, subdir)
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def __gcs_cache(self, subdir: str, **kwargs) -> str:
        """
        Computes the absolute path of a subdirectory of the cache directory in the GCS bucket used to persist the cache.

        Args:
            subdir: Name of the cache subdirectory.

        Returns:
            Absolute path of the cache subdirectory.
        """

        for key in kwargs.values():
            subdir += f"_{key}"
        return FileManager.ensure_trailing_slash(os.path.join(self.gcs_cache_dir, subdir))

    def cached_file_path(self, subdir: str, file_path: str, **kwargs) -> str:
        """
        Calculates the absolute path of a file in the cache.

        Args:
            subdir: Name of the cache subdirectory.
            file_path: File path.

        Returns:
            Absolute path of the file in cache.
        """

        file_name = os.path.basename(file_path)
        return os.path.join(self.cache(subdir, **kwargs), file_name)

    def data_folder(self, dataset: str, subdir: str = "", **kwargs) -> str:
        """
        Computes the absolute path of a subdirectory of a dataset directory. Creates all subdirectories of this path
        that do not exist yet.

        Args:
            dataset: dataset: Name of a dataset (e.g. "train", "val", "test").
            subdir: Name of the subdirectory (e.g. "audio", "spectrograms").

        Returns:
            Absolute path of the subdirectory.
        """

        if subdir == "spectrograms":
            for key in kwargs.values():
                subdir += f"_{key}"
        data_folder_path = os.path.join(self.base_dir, dataset, subdir)
        os.makedirs(data_folder_path, exist_ok=True)
        return data_folder_path

    def model_dir(self) -> str:
        """

        Returns:
            Absolute path of the directory that is used to store model files.
        """

        return os.path.join(self.base_dir, "models")

    def gcs_model_dir(self) -> str:
        """

        Returns:
            Path of the directory within the GCS bucket that is used to persist model files.
        """

        return self.gcs_model_dir

    def copy_cache_to_gcs(self, subdir: str, **kwargs) -> None:
        """
        Uploads the contents of a cache subdirectory to the GCS bucket used to persist the cache.

        Args:
            subdir: Name of the cache subdirectory.

        Returns:
            None
        """

        FileManager.gcs_copy_dir(self.cache(subdir), self.__gcs_cache(subdir, **kwargs))

    def copy_cache_from_gcs(self, subdir: str, **kwargs) -> None:
        """
        Downloads the contents of a cache subdirectory from the GCS bucket used to persist the cache.

        Args:
            subdir: Name of the cache subdirectory.

        Returns:
            None
        """

        if self.gcs_file_exists(self.__gcs_cache(subdir, **kwargs)):
            FileManager.gcs_copy_dir(self.__gcs_cache(subdir, **kwargs), self.cache(subdir, **kwargs))

    def copy_file_to_gcs_cache(self, file_path: Union[str, List[str]], subdir: str, **kwargs) -> None:
        """
        Uploads the specified file to a cache subdirectory in the GCS bucket used to persist the cache.

        Args:
            file_path: Path of the file to be uploaded.
            subdir: Name of the cache subdirectory.

        Returns:
            None
        """

        if type(file_path) == list:
            FileManager.gcs_copy_files(file_path, self.__gcs_cache(subdir, **kwargs))
        else:
            FileManager.gcs_copy_file(file_path, self.__gcs_cache(subdir, **kwargs))

    def clear_cache(self, subdir: str, **kwargs) -> None:
        """
        Empties the specified subdirectory of the cache directory. If the cache is persisted to a GCS bucket, the
        corresponding folder in the GCS bucket is also emptied.

        Args:
            subdir: Name of the cache subdirectory.

        Returns:
            None
        """

        FileManager.empty_dir(self.cache(subdir, **kwargs))
        if self.is_pipeline_run:
            FileManager.gcs_remove_dir(self.__gcs_cache(subdir, **kwargs))
