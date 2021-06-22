import logging
import torch
import os

from data_preparation import filepaths, downloader, spectrograms
from training import training, dataset

from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from general.logging import logger

class PipelineSteps:
    def download_xeno_canto_data(self, gcs_bucket: str, output_path: str, verbose_logging: bool, **kwargs):
        if verbose_logging:
            logger.setLevel(logging.VERBOSE)
        path_manager = filepaths.PathManager(output_path, gcs_bucket=gcs_bucket)
        xc_downloader = downloader.XenoCantoDownloader(path_manager)
        xc_downloader.create_datasets(**kwargs)

    def create_spectrograms(self, input_path: str, gcs_bucket: str, output_path: str, chunk_length: int,
                            clear_spectrogram_cache: bool = False, verbose_logging: bool = False):
        if verbose_logging:
            logger.setLevel(logging.VERBOSE)
        audio_path_manager = filepaths.PathManager(
            input_path, gcs_bucket=gcs_bucket)
        spectrogram_path_manager = filepaths.PathManager(
            output_path, gcs_bucket=gcs_bucket)
        spectrogram_creator = spectrograms.SpectrogramCreator(
            chunk_length, audio_path_manager, spectrogram_path_manager)

        spectrogram_creator.create_spectrograms_for_splits(
            splits=["train", "val", "test"], clear_spectrogram_cache=clear_spectrogram_cache)

    def train_model(self, input_path: str, gcs_bucket: str, output_path: str, chunk_length: int, batch_size: int = 32, number_epochs: int = 25):
        spectrogram_path_manager = filepaths.PathManager(input_path, gcs_bucket=gcs_bucket)
        train_set = dataset.XenoCantoSpectrograms(
            spectrogram_path_manager, chunk_length=chunk_length, split="train")
        train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=0)

        val_set = dataset.XenoCantoSpectrograms(
            spectrogram_path_manager, chunk_length=chunk_length, split="val")
        val_loader = DataLoader(
            val_set, batch_size=batch_size, shuffle=True, num_workers=0)

        test_set = dataset.XenoCantoSpectrograms(
            spectrogram_path_manager, chunk_length=chunk_length, split="test")
        test_loader = DataLoader(
            test_set, batch_size=batch_size, shuffle=True, num_workers=0)

        resnet_model = models.resnet18(pretrained=True)
        number_classes = len(train_loader.dataset.class_to_idx)
        num_ftrs = resnet_model.fc.in_features
        resnet_model.fc = nn.Linear(num_ftrs, number_classes)

        my_criterion = nn.CrossEntropyLoss()
        my_optimizer = optim.SGD(resnet_model.parameters(), lr=0.001, momentum=0.9)

        exp_lr_scheduler = lr_scheduler.StepLR(my_optimizer, step_size=7, gamma=0.1)

        model = training.train_model(train_loader, val_loader, test_loader, resnet_model, my_criterion,
                                        my_optimizer, exp_lr_scheduler, number_epochs, number_classes=number_classes)


        filepaths.PathManager.ensure_dir(output_path)

        torch.save(model, os.path.join(output_path, "model.pt"))
