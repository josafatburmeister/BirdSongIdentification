import copy
import os
from datetime import datetime
from typing import Dict

import torch

from general import logger, FileManager
from training import metrics


class ModelTracker:
    """
    Tracks model performance across training epochs and saves the models with the best performance.
    """

    def __init__(self, pathmanager: FileManager, experiment_name: str, id_to_class_mapping: Dict[str, str],
                 is_pipeline_run: bool, model: torch.nn.Module, multi_label_classification: bool = True,
                 device: torch.device = torch.device('cpu')) -> None:
        """

        Args:
            pathmanager: FileManager object that manages the directory containing the spectrograms file and
                their labels.
            experiment_name: Descriptive name of the experiment.
            id_to_class_mapping: Dictionary that maps class indices to human-readable class names.
            is_pipeline_run: Whether the model training is run in a non-notebook execution environment.
            model: A Pytorch model.
            multi_label_classification: Whether the model is trained as single-label classification model or as
                multi-label classification model.
            device: Pytorch device that should be used for metric calculations.
        """
        self.path = pathmanager
        self.experiment_name = experiment_name
        self.id_to_class_mapping = id_to_class_mapping
        self.is_pipeline_run = is_pipeline_run
        self.num_classes = len(self.id_to_class_mapping)
        self.multi_label_classification = multi_label_classification
        self.best_average_epoch = 1
        self.best_average_metrics = metrics.Metrics(num_classes=self.num_classes,
                                                    multi_label=self.multi_label_classification,
                                                    device=device)
        self.best_average_model = copy.deepcopy(model.state_dict())
        self.best_minimum_epoch = 1
        self.best_minimum_metrics = metrics.Metrics(num_classes=self.num_classes,
                                                    multi_label=self.multi_label_classification,
                                                    device=device)
        self.best_minimum_model = copy.deepcopy(model.state_dict())
        self.best_epochs_per_class = {}
        self.best_metrics_per_class = {}
        self.best_models_per_class = {}
        for class_name in self.id_to_class_mapping.values():
            self.best_epochs_per_class[class_name] = 1
            self.best_metrics_per_class[class_name] = metrics.Metrics(num_classes=self.num_classes,
                                                                      multi_label=self.multi_label_classification,
                                                                      device=device)
            self.best_models_per_class[class_name] = copy.deepcopy(model.state_dict())

    def track_best_model(self, model: torch.nn.Module, model_metrics: metrics.Metrics, epoch: int) -> None:
        """
        Tracks the performance metrics of the current training epoch and updates the best model if necessary.

        Args:
            model: Pytorch model obtained from the current training epoch.
            model_metrics: Metrics object of current training epoch.
            epoch: Current training epoch.

        Returns:
            None
        """

        model_mean_f1 = torch.mean(model_metrics.f1_score())
        logger.info(f"Average F1-score of current epoch: {model_mean_f1}")
        if torch.mean(self.best_average_metrics.f1_score()) < model_mean_f1:
            self.best_average_epoch = epoch + 1
            self.best_average_metrics = copy.deepcopy(model_metrics)
            self.best_average_model = copy.deepcopy(model.state_dict())

        if torch.min(self.best_minimum_metrics.f1_score()) < torch.min(model_metrics.f1_score()) \
                or torch.min(self.best_minimum_metrics.f1_score()) == torch.min(model_metrics.f1_score()) \
                and torch.mean(self.best_minimum_metrics.f1_score()) < model_mean_f1:
            self.best_minimum_epoch = epoch + 1
            self.best_minimum_metrics = copy.deepcopy(model_metrics)
            self.best_minimum_model = copy.deepcopy(model.state_dict())

        if self.multi_label_classification:
            for class_id, class_name in self.id_to_class_mapping.items():
                best_f1 = self.best_metrics_per_class[class_name].f1_score()[class_id].item()
                model_f1 = model_metrics.f1_score()[class_id].item()
                best_mean_f1 = torch.mean(self.best_metrics_per_class[class_name].f1_score())
                if best_f1 < model_f1 or (best_f1 == model_f1 and best_mean_f1 < model_mean_f1):
                    self.best_epochs_per_class[class_name] = epoch + 1
                    self.best_metrics_per_class[class_name] = copy.deepcopy(model_metrics)
                    self.best_models_per_class[class_name] = copy.deepcopy(model.state_dict())
        logger.info(f"Best average F1-score: {torch.mean(self.best_average_metrics.f1_score())}")

    def __save_model(self, model, model_path: str) -> None:
        """
        Saves model to a file.

        Args:
            model: Pytorch model.
            model_path: Path where the model is to be saved.

        Returns:
            None
        """
        os.makedirs(os.path.split(model_path)[0], exist_ok=True)
        torch.save({
            'state_dict': model,
            'id_to_class_mapping': self.id_to_class_mapping
        }, model_path)

        if self.is_pipeline_run:
            gcs_model_path = os.path.join(self.path.gcs_model_dir(), model_path.lstrip(self.path.model_dir()))
            self.path.gcs_copy_file(model_path, gcs_model_path)

    def save_epoch_model(self, model: torch.nn.Module, epoch: int, run_id: str) -> None:
        """
        Saves model obtained from current training epoch to a file and optionally to Weights and Biases.

        Args:
            model: Pytorch model obtained from the current training epoch.
            epoch: Current training epoch.
            run_id: Weights and Biases run ID.

        Returns:
            None
        """

        current_time_str = datetime.now().strftime("%Y%m%d-%H.%M.%S")
        model_name = f"{self.experiment_name}_{current_time_str}_epoch{str(epoch).zfill(2)}.pt"
        model_path = os.path.join(self.path.model_dir(), self.experiment_name, model_name)
        if run_id:
            model_path.replace(".pt", f"_{run_id}.pt")
        self.__save_model(model, model_path)

    def save_best_models(self, run_id: str) -> None:
        """
        Saves best models to files and optionally to Weights and Biases.

        Args:
            run_id: Weights and Biases run ID.

        Returns:
            None
        """

        current_time_str = datetime.now().strftime("%Y%m%d-%H.%M.%S")
        for model, model_name in [(self.best_average_model, f"{self.experiment_name}_avg_model_{current_time_str}.pt"),
                                  (self.best_minimum_model, f"{self.experiment_name}_min_model_{current_time_str}.pt")]:
            model_path = os.path.join(self.path.model_dir(), self.experiment_name, model_name)
            if run_id:
                model_path.replace(".pt", f"_{run_id}.pt")
            self.__save_model(model, model_path)

        if self.multi_label_classification:
            for class_name, model in self.best_models_per_class.items():
                model_path = os.path.join(self.path.model_dir(), self.experiment_name,
                                          f"{self.experiment_name}_{class_name}_{current_time_str}.pt")
                if run_id:
                    model_path.replace(".pt", f"_{run_id}.pt")
                self.__save_model(model, model_path)
