import logging
import subprocess

import joblib


def gcs_copy(src_path, dst_path) -> None:
    logging.info(
        subprocess.run(['gsutil', 'cp', src_path, dst_path], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8'))
    logging.info(f'Copied {src_path} to {dst_path}')


def gcs_copy_dir(src_path, dst_path) -> None:
    logging.info(
        subprocess.run(['gsutil', 'cp', '-r', src_path, dst_path], stdout=subprocess.PIPE).stdout[:-1].decode('utf-8'))
    logging.info(f'Copied {src_path} to {dst_path}')


def save_model(model, model_file) -> None:
    """Save XGBoost model for serving."""
    joblib.dump(model, model_file)
    logging.info("Model export success: %s", model_file)
