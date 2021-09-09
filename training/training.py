import time
from typing import Dict, Tuple, Union

import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from general.logging import logger
from models import densenet, resnet
from training import dataset as ds, metrics, metric_logging, model_tracker as tracker
from training.early_stopper import EarlyStopper


class ModelTrainer:
    """
    Runs model training with fixed hyperparameter values.
    """

    @staticmethod
    def __setup_device() -> torch.device:
        """
        Creates the Pytorch device on which the model training is to be run; if a GPU with CUDA support is available,
        it is preferred over the CPU.

        Returns:
            Pytorch device.
        """

        device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info('Device set to: %s', device)
        return torch.device(device)

    def __init__(self,
                 spectrogram_path_manager,
                 architecture: str,
                 experiment_name: str,
                 batch_size: int = 100,
                 early_stopping=False,
                 include_noise_samples=True,
                 is_hyperparameter_tuning=False,
                 layers_to_unfreeze="all",
                 learning_rate=0.001,
                 learning_rate_scheduler=None,
                 learning_rate_scheduler_gamma=0.1,
                 learning_rate_scheduler_step_size=7,
                 momentum=0.9,
                 multi_label_classification=True,
                 multi_label_classification_threshold=0.5,
                 number_epochs=25,
                 number_workers=0,
                 optimizer="Adam",
                 p_dropout=0,
                 save_all_models=False,
                 track_metrics=True,
                 train_dataset="train",
                 monitor="f1_score",
                 patience=0,
                 min_change=0.0,
                 undersample_noise_samples=True,
                 val_dataset="val",
                 wandb_entity_name="",
                 wandb_key="",
                 wandb_project_name="",
                 weight_decay=0) -> None:
        """

        Args:
            spectrogram_path_manager: FileManager object that manages the directory containing the spectrograms file and
                their labels.
            architecture: Model architecture, either "resnet18", "resnet34", "resnet50", or "densenet121".
            experiment_name: Descriptive name of the training run / experiment.
            batch_size: Batch size.
            early_stopping: Whether early stopping should be used for model training.
            include_noise_samples: Whether spectrograms that are classified as "noise" during noise filtering should be
                included in the spectrogram dataset.
            is_hyperparameter_tuning: Whether the training run is part of a hyperparameter tuning.
            layers_to_unfreeze: List of model layer names to be unfrozen for fine-tuning; if set to "all", all model
                layers will be fine-tuned.
            learning_rate: Learning rate.
            learning_rate_scheduler: Learning rate scheduler, either "cosine" (cosine annealing learning rate scheduler)
                or "step_lr" (step-wise learning rate decay).
            learning_rate_scheduler_gamma: Gamma parameter of step-wise learning rate decay; only considered if
                "learning_rate_scheduler" is set to "step_lr".
            learning_rate_scheduler_step_size: Step size of step-wise learning rate decay; only considered if
                "learning_rate_scheduler" is set to "step_lr".
            momentum: Momentum; only considered if "optimizer" is set to "SGD".
            multi_label_classification: Whether the model should be trained as single-label classification model or as
                multi-label classification model.
            multi_label_classification_threshold: Threshold for assigning samples to positive class in multi-label
                classification, only considered if "multi_label_classification" is set to True.
            number_epochs: Number of training epochs.
            number_workers: Number of dataloading workers.
            optimizer: Optimizer, either "Adam" or "SGD".
            p_dropout: Probability of dropout before the fully-connected layer.
            save_all_models: Whether the models from all training epochs should be saved or only the models with the
                best performance.
            track_metrics: Whether the model metrics should be logged in Weights and Biases (https://wandb.ai/);
                requires "wandb_entity_name", "wandb_key", and "wandb_project_name" to be set.
            train_dataset: Name of the dataset to be used for model training.
            monitor: Name of the metric that should be used for early stopping; only considered if "early_stopping" is
                set to True.
            patience: Patience parameter for early stopping, only considered if "early_stopping" is True.
            min_change: Minimum change parameter for early stopping, only considered if "early_stopping" is True.
            undersample_noise_samples: Whether the number of "noise" spectrograms should be limited to the maximum
                number of spectrograms of a sound class.
            val_dataset: Name of the dataset to be used for model validation.
            wandb_entity_name: Name of the Weights and Biases account to which the model metrics should be logged; only
                considered if "track_metrics" is set to True.
            wandb_key: API key for the Weights and Biases account to which the model metrics should be logged; only
                considered if "track_metrics" is set to True.
            wandb_project_name: Name of the Weights and Biases project to which the model metrics should be logged; only
                considered if "track_metrics" is set to True.
            weight_decay: Weight decay.
        """

        self.spectrogram_path_manager = spectrogram_path_manager
        self.architecture = architecture
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.experiment_name = experiment_name
        self.include_noise_samples = include_noise_samples
        self.is_hyperparameter_tuning = is_hyperparameter_tuning
        self.layers_to_unfreeze = layers_to_unfreeze
        self.learning_rate = learning_rate
        self.learning_rate_scheduler = learning_rate_scheduler
        self.learning_rate_scheduler_gamma = learning_rate_scheduler_gamma
        self.learning_rate_scheduler_step_size = learning_rate_scheduler_step_size
        self.momentum = momentum
        self.multi_label_classification = multi_label_classification
        self.multi_label_classification_threshold = multi_label_classification_threshold
        self.num_epochs = number_epochs
        self.num_workers = number_workers
        self.optimizer = optimizer
        self.p_dropout = p_dropout
        self.save_all_models = save_all_models
        self.train_dataset = train_dataset
        self.track_metrics = track_metrics
        self.undersample_noise_samples = undersample_noise_samples
        self.val_dataset = val_dataset
        self.weight_decay = weight_decay

        # early stopping parameters
        self.monitor = monitor
        self.patience = patience
        self.min_change = min_change

        # setup datasets
        datasets, dataloaders = self.__setup_dataloaders()
        self.datasets = datasets
        self.dataloaders = dataloaders
        self.num_classes = self.datasets["train"].num_classes()

        self.is_pipeline_run = self.spectrogram_path_manager.is_pipeline_run

        config = locals().copy()
        del config['spectrogram_path_manager']
        device = self.__setup_device()
        self.logger = metric_logging.TrainingLogger(self, config, self.is_pipeline_run, track_metrics=track_metrics,
                                                    wandb_entity_name=wandb_entity_name,
                                                    wandb_project_name=wandb_project_name, wandb_key=wandb_key)

    def __setup_dataloaders(self) -> Tuple[Dict[str, ds.SpectrogramDataset], Dict[str, torch.utils.data.DataLoader]]:
        """
        Creates datasets and dataloaders for model training and validation.

        Returns:
            Training and validation dataset and training and validation dataloader.
        """

        datasets = {}
        dataloaders = {}

        for dataset in [self.train_dataset, self.val_dataset]:
            dataset_name = "train" if dataset == self.train_dataset else "val"
            datasets[dataset_name] = ds.SpectrogramDataset(
                self.spectrogram_path_manager,
                include_noise_samples=self.include_noise_samples, dataset=dataset,
                multi_label_classification=self.multi_label_classification,
                undersample_noise_samples=self.undersample_noise_samples)

            shuffle = (dataset == self.train_dataset)
            dataloaders[dataset_name] = DataLoader(
                datasets[dataset_name], batch_size=self.batch_size, sampler=None,
                shuffle=shuffle, num_workers=self.num_workers)

        return datasets, dataloaders

    def __setup_model(self) -> torch.nn.Module:
        """
        Creates model with the required number of classes and the required architecture.

        Returns:
            Pytorch model.
        """

        logger.info("Setup %s model: ", self.architecture)
        model = None
        if self.architecture == "resnet18":
            model = resnet.ResnetTransferLearning(architecture="resnet18",
                                                  num_classes=self.num_classes,
                                                  layers_to_unfreeze=self.layers_to_unfreeze,
                                                  logger=logger,
                                                  p_dropout=self.p_dropout)
        if self.architecture == "resnet34":
            model = resnet.ResnetTransferLearning(architecture="resnet34",
                                                  num_classes=self.num_classes,
                                                  layers_to_unfreeze=self.layers_to_unfreeze,
                                                  logger=logger,
                                                  p_dropout=self.p_dropout)
        elif self.architecture == "resnet50":
            model = resnet.ResnetTransferLearning(architecture="resnet50",
                                                  num_classes=self.num_classes,
                                                  layers_to_unfreeze=self.layers_to_unfreeze,
                                                  logger=logger,
                                                  p_dropout=self.p_dropout)

        elif self.architecture == "densenet121":
            model = densenet.DenseNet121TransferLearning(
                num_classes=self.num_classes, layers_to_unfreeze=self.layers_to_unfreeze,
                logger=logger, p_dropout=self.p_dropout)

        logger.info("\n")
        model.id_to_class_mapping = self.datasets["train"].id_to_class_mapping(
        )
        return model

    def __setup_optimization(self, model: torch.nn.Module) -> Tuple[
        torch.nn._Loss, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """
        Sets up tools for model optimization.

        Args:
            model: Pytorch model.

        Returns:
            Pytorch loss, optimizer and learning rate scheduler.
        """

        if self.multi_label_classification:
            loss = torch.nn.BCEWithLogitsLoss()
        else:
            loss = torch.nn.CrossEntropyLoss()
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(), self.learning_rate, weight_decay=self.weight_decay
            )
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(), self.learning_rate, self.momentum, weight_decay=self.weight_decay
            )
        else:
            optimizer = None
        if self.learning_rate_scheduler == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.num_epochs, eta_min=0)
        elif self.learning_rate_scheduler == "step_lr":
            scheduler = lr_scheduler.StepLR(optimizer, step_size=self.learning_rate_scheduler_step_size,
                                            gamma=self.learning_rate_scheduler_gamma)
        else:
            scheduler = None
        return loss, optimizer, scheduler

    def train_model(self) -> Union[Tuple[torch.nn.Module, torch.nn.Module, Dict[str, torch.nn.Module]], float]:
        """
        Model training loop.

        Returns:
            Models with the best performance (if is_hyperparameter_tuning is False) or macro-averaged F1-score of the
            best model.
        """

        model = self.__setup_model()
        device = ModelTrainer.__setup_device()
        loss_function, optimizer, scheduler = self.__setup_optimization(model)

        model_tracker = tracker.ModelTracker(self.spectrogram_path_manager,
                                             self.experiment_name,
                                             self.datasets["train"].id_to_class_mapping(
                                             ),
                                             self.is_pipeline_run, model,
                                             self.multi_label_classification,
                                             device)

        model.to(device)

        since = time.time()

        logger.info("Number of species: %i", self.num_classes)

        if self.early_stopping:
            early_stopper = EarlyStopper(
                monitor=self.monitor, patience=self.patience, min_change=self.min_change)

        logger.info("\n")
        for epoch in range(self.num_epochs):
            logger.info("Epoch %i/%i", epoch + 1, self.num_epochs)
            logger.info("----------")

            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                model_metrics = metrics.Metrics(num_classes=self.num_classes,
                                                multi_label=self.multi_label_classification,
                                                device=device)

                for images, labels in self.dataloaders[phase]:
                    images = images.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(images)
                        if self.multi_label_classification:
                            predictions = (torch.sigmoid(
                                outputs) > self.multi_label_classification_threshold).int()
                        else:
                            _, predictions = torch.max(outputs, 1)
                        loss = loss_function(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        if self.learning_rate_scheduler:
                            scheduler.step()

                    model_metrics.update(predictions, labels, loss)

                if phase == "val":
                    model_tracker.track_best_model(model, model_metrics, epoch)

                self.logger.log_metrics(
                    model_metrics, phase, epoch, loss if phase == "train" else None)

                if phase == "val" and self.save_all_models:
                    model_tracker.save_epoch_model(
                        model, epoch, self.logger.get_run_id())

            if self.early_stopping and early_stopper.check_early_stopping(model_metrics):
                logger.info(f"Training stopped early, because {self.monitor}, did not improve by at least"
                            f"{self.min_change} for the last {self.patience} epochs.")
                break

        time_elapsed = time.time() - since
        logger.info("Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60))

        if not self.save_all_models:
            model_tracker.save_best_models(self.logger.get_run_id())

        self.logger.print_model_summary(
            model_tracker.best_average_epoch,
            model_tracker.best_average_metrics,
            model_tracker.best_minimum_epoch,
            model_tracker.best_minimum_metrics,
            model_tracker.best_epochs_per_class if self.multi_label_classification else None,
            model_tracker.best_metrics_per_class if self.multi_label_classification else None
        )

        if self.is_pipeline_run:
            metric_logging.TrainingLogger.log_metrics_in_kubeflow(
                model_tracker.best_average_metrics,
                model_tracker.best_minimum_metrics,
            )

        self.logger.finish()

        model.load_state_dict(model_tracker.best_average_model)
        best_average_model = model
        model.load_state_dict(model_tracker.best_minimum_model)
        best_minimum_model = model
        best_models_per_class = {}
        for class_name, state_dict in model_tracker.best_models_per_class.items():
            model.load_state_dict(state_dict)
            best_models_per_class[class_name] = model

        if self.is_hyperparameter_tuning:
            return torch.mean(model_tracker.best_average_metrics.f1_score()).item()
        return best_average_model, best_minimum_model, best_models_per_class
