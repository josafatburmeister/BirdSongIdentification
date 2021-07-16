import time

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from training.early_stopper import EarlyStopper

from general.logging import logger
from training import dataset, metrics, metric_logging, model_tracker as tracker
from models import densenet, resnet

class ModelTrainer:
    @staticmethod
    def setup_device() -> torch.device:
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info('Device set to: %s', device)
        return torch.device(device)

    def __init__(self,
                 spectrogram_path_manager,
                 architecture: str,
                 chunk_length: int,
                 experiment_name: str,
                 batch_size: int = 100,
                 include_noise_samples=True,
                 is_hyperparameter_tuning=False,
                 layers_to_unfreeze=None,
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
                 track_metrics=True,
                 monitor="f1_score",
                 patience=0,
                 min_change=0.0,
                 undersample_noise_samples=True,
                 wandb_entity_name="",
                 wandb_key="",
                 wandb_project_name=""):
        self.spectrogram_path_manager = spectrogram_path_manager
        self.architecture = architecture
        self.batch_size = batch_size
        self.chunk_length = chunk_length
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
        self.track_metrics = track_metrics
        self.undersample_noise_samples = undersample_noise_samples

        # early stopping parameters
        self.monitor = monitor
        self.patience = patience
        self.min_change = min_change

        # setup datasets
        datasets, dataloaders = self.setup_dataloaders()
        self.datasets = datasets
        self.dataloaders = dataloaders
        self.num_classes = self.datasets["train"].num_classes()

        self.is_pipeline_run = self.spectrogram_path_manager.is_pipeline_run

        config = locals().copy()
        del config['spectrogram_path_manager']
        device = self.setup_device()
        self.logger = metric_logging.TrainingLogger(self, config, self.is_pipeline_run, track_metrics=track_metrics,
                                                    wandb_entity_name=wandb_entity_name,
                                                    wandb_project_name=wandb_project_name, wandb_key=wandb_key)

    def setup_dataloaders(self):
        datasets = {}
        dataloaders = {}

        for split in ["train", "val"]:
            datasets[split] = dataset.XenoCantoSpectrograms(
                self.spectrogram_path_manager, chunk_length=self.chunk_length,
                include_noise_samples=self.include_noise_samples, split=split,
                multi_label_classification=self.multi_label_classification,
                undersample_noise_samples=self.undersample_noise_samples)

            shuffle = (split == "train")
            dataloaders[split] = DataLoader(
                datasets[split], batch_size=self.batch_size, sampler=None,
                shuffle=shuffle, num_workers=self.num_workers)

        return datasets, dataloaders

    def setup_model(self):
        logger.info("Setup %s model: ", self.architecture)
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
                                                  logger=self.logger.get(),
                                                  p_dropout=self.p_dropout)
        elif self.architecture == "resnet50":
            model = resnet.ResnetTransferLearning(architecture="resnet50",
                                                  num_classes=self.num_classes,
                                                  layers_to_unfreeze=self.layers_to_unfreeze,
                                                  logger=self.logger.get(),
                                                  p_dropout=self.p_dropout)

        elif self.architecture == "densenet121":
            model = densenet.DenseNet121TransferLearning(
                num_classes=self.num_classes, layers_to_unfreeze=self.layers_to_unfreeze,
                logger=self.logger.get(), p_dropout=self.p_dropout)

        logger.info("\n")
        model.id_to_class_mapping = self.datasets["train"].id_to_class_mapping()
        return model

    def setup_optimization(self, model):
        if self.multi_label_classification:
            loss = torch.nn.BCEWithLogitsLoss()
        else:
            loss = torch.nn.CrossEntropyLoss()
        if self.optimizer == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(), self.learning_rate
            )
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(), self.learning_rate, self.momentum
            )
        else:
            optimizer = None
        if self.learning_rate_scheduler == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=0)
        elif self.learning_rate_scheduler == "step_lr":
            scheduler = lr_scheduler.StepLR(optimizer, step_size=self.learning_rate_scheduler_step_size,
                                            gamma=self.learning_rate_scheduler_gamma)
        else:
            scheduler = None
        return loss, optimizer, scheduler

    def train_model(self):
        model = self.setup_model()
        device = ModelTrainer.setup_device()
        loss_function, optimizer, scheduler = self.setup_optimization(model)

        model_tracker = tracker.ModelTracker(self.spectrogram_path_manager,
                                             self.experiment_name,
                                             self.datasets["train"].id_to_class_mapping(),
                                             self.is_pipeline_run, model,
                                             self.multi_label_classification,
                                             device)

        model.to(device)

        since = time.time()

        logger.info("Number of species: %i", self.num_classes)

        early_stopper = EarlyStopper(monitor=self.monitor, patience=self.patience, min_change=self.min_change)

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
                            predictions = (torch.sigmoid(outputs) > self.multi_label_classification_threshold).int()
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

                self.logger.log_metrics(model_metrics, phase, epoch, loss if phase == "train" else None)

            if early_stopper.check_early_stopping(model_metrics):
                logger.info(f"Training stopped early, because {self.monitor}, did not improve by at least"
                            f"{self.min_change} for the last {self.patience} epochs.")
                break

        time_elapsed = time.time() - since
        logger.info("Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60))

        model_tracker.save_best_models(self.logger.get_run_id())

        if not self.is_hyperparameter_tuning:
            self.logger.print_model_summary(
                model_tracker.best_average_epoch,
                model_tracker.best_average_metrics,
                model_tracker.best_minimum_epoch,
                model_tracker.best_minimum_metrics,
                model_tracker.best_epochs_per_class if self.multi_label_classification else None,
                model_tracker.best_metrics_per_class if self.multi_label_classification else None
            )

        if self.is_pipeline_run:
            self.logger.log_metrics_in_kubeflow(
                model_tracker.best_average_metrics,
                model_tracker.best_minimum_metrics,
                model_tracker.best_metrics_per_class if self.multi_label_classification else None
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
            return model_tracker.best_average_metrics.f1_score()
        return best_average_model, best_minimum_model, best_models_per_class
