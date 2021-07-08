import copy
import os
import torch
from datetime import datetime
from data_preparation import filepaths
from training import metrics
from general.logging import logger


class ModelTracker:
    def __init__(self, pathmanager: filepaths.PathManager, experiment_name: str, id_to_class_mapping: dict,
                 is_pipeline_run: bool, model: torch.nn.Module, multi_label_classification: bool = True):
        self.path = pathmanager
        self.experiment_name = experiment_name
        self.id_to_class_mapping = id_to_class_mapping
        self.is_pipeline_run = is_pipeline_run
        self.num_classes = len(self.id_to_class_mapping)
        self.multi_label_classification = multi_label_classification
        self.best_average_epoch = 1
        self.best_average_metrics = metrics.Metrics(num_classes=self.num_classes,
                                                    multi_label=self.multi_label_classification)
        self.best_average_model = copy.deepcopy(model.state_dict())
        self.best_minimum_epoch = 1
        self.best_minimum_metrics = metrics.Metrics(num_classes=self.num_classes,
                                                    multi_label=self.multi_label_classification)
        self.best_minimum_model = copy.deepcopy(model.state_dict())
        self.best_epochs_per_class = {}
        self.best_metrics_per_class = {}
        self.best_models_per_class = {}
        for class_name in self.id_to_class_mapping.values():
            self.best_epochs_per_class[class_name] = 1
            self.best_metrics_per_class[class_name] = metrics.Metrics(num_classes=self.num_classes,
                                                                      multi_label=self.multi_label_classification)
            self.best_models_per_class[class_name] = copy.deepcopy(model.state_dict())

    def track_best_model(self, model: torch.nn.Module, model_metrics: metrics.Metrics, epoch: int):
        if torch.mean(self.best_average_metrics.f1_score()) < torch.mean(model_metrics.f1_score()):
            self.best_average_epoch = epoch + 1
            self.best_average_metrics = model_metrics
            self.best_average_model = copy.deepcopy(model.state_dict())

        if torch.min(self.best_minimum_metrics.f1_score()) < torch.min(model_metrics.f1_score()) \
                or torch.min(self.best_minimum_metrics.f1_score()) == torch.min(model_metrics.f1_score()) \
                and torch.mean(self.best_minimum_metrics.f1_score()) < torch.mean(model_metrics.f1_score()):
            self.best_minimum_epoch = epoch + 1
            self.best_minimum_metrics = model_metrics
            self.best_minimum_model = copy.deepcopy(model.state_dict())

        if self.multi_label_classification:
            for class_id, class_name in self.id_to_class_mapping.items():
                if self.best_metrics_per_class[class_name].f1_score()[class_id].item() < model_metrics.f1_score()[
                    class_id].item() \
                        or self.best_metrics_per_class[class_name].f1_score()[class_id].item() == \
                        model_metrics.f1_score()[
                            class_id].item() \
                        and torch.mean(self.best_metrics_per_class[class_name].f1_score()).item() < torch.mean(
                    model_metrics.f1_score()).item():
                    self.best_epochs_per_class[class_name] = epoch + 1
                    self.best_metrics_per_class[class_name] = model_metrics
                    self.best_models_per_class[class_name] = copy.deepcopy(model.state_dict())

    def save_model(self, model, model_path: str):
        self.path.ensure_dir(os.path.split(model_path)[0])
        torch.save({
            'state_dict': model,
            'id_to_class_mapping': self.id_to_class_mapping
        }, model_path)

        if self.is_pipeline_run:
            gcs_model_path = os.path.join(self.path.gcs_model_dir(), model_path.lstrip(self.path.model_dir()))
            self.path.gcs_copy_file(model_path, gcs_model_path)

    def save_best_models(self, run_id: str):
        current_time_str = datetime.now().strftime("%Y%m%d-%H.%M.%S")
        for model, model_name in [(self.best_average_model, f"{self.experiment_name}_avg_model_{current_time_str}.pt"),
                                  (self.best_minimum_model, f"{self.experiment_name}_min_model_{current_time_str}.pt")]:
            model_path = os.path.join(self.path.model_dir(), self.experiment_name, model_name)
            if run_id:
                model_path.replace(".pt", f"_{run_id}.pt")
            self.save_model(model, model_path)

        if self.multi_label_classification:
            for class_name, model in self.best_models_per_class.items():
                model_path = os.path.join(self.path.model_dir(), self.experiment_name,
                                          f"{self.experiment_name}_{class_name}_{current_time_str}.pt")
                if run_id:
                    model_path.replace(".pt", f"_{run_id}.pt")
                self.save_model(model, model_path)
