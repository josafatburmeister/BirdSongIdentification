import torch
import time
from training import dataset, metrics, metric_logging, model_tracker as tracker
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn
from general.logging import logger


class ModelTrainer:
    @staticmethod
    def setup_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(self,
                 spectrogram_path_manager,
                 architecture: str,
                 chunk_length: int,
                 experiment_name: str,
                 batch_size: int = 100,
                 include_noise_samples=True,
                 layers_to_unfreeze=None,
                 learning_rate=0.001,
                 learning_rate_scheduler=None,
                 learning_rate_scheduler_gamma=0.1,
                 learning_rate_scheduler_step_size=7,
                 multi_label_classification=True,
                 multi_label_classification_threshold=0.5,
                 number_epochs=25,
                 number_workers=0,
                 optimizer="Adam",
                 track_metrics=True,
                 wandb_entity_name="",
                 wandb_key="",
                 wandb_project_name=""):
        self.spectrogram_path_manager = spectrogram_path_manager
        self.architecture = architecture
        self.batch_size = batch_size
        self.chunk_length = chunk_length
        self.experiment_name = experiment_name
        self.include_noise_samples = include_noise_samples
        self.layers_to_unfreeze = layers_to_unfreeze
        self.learning_rate = learning_rate
        self.learning_rate_scheduler = learning_rate_scheduler
        self.learning_rate_scheduler_gamma = learning_rate_scheduler_gamma
        self.learning_rate_scheduler_step_size = learning_rate_scheduler_step_size
        self.multi_label_classification = multi_label_classification
        self.multi_label_classification_threshold = multi_label_classification_threshold
        self.num_epochs = number_epochs
        self.num_workers = number_workers
        self.optimizer = optimizer
        self.track_metrics = track_metrics

        # setup datasets
        datasets, dataloaders = self.setup_dataloaders()
        self.datasets = datasets
        self.dataloaders = dataloaders
        self.num_classes = self.datasets["train"].num_classes()

        self.is_pipeline_run = self.spectrogram_path_manager.is_pipeline_run

        config = locals().copy()
        del config['spectrogram_path_manager']
        self.logger = metric_logging.TrainingLogger(self, config, self.is_pipeline_run, track_metrics=track_metrics,
                                                    wandb_entity_name=wandb_entity_name,
                                                    wandb_project_name=wandb_project_name, wandb_key=wandb_key)

    def setup_dataloaders(self):
        datasets = {}
        dataloaders = {}

        for split in ["train", "val", "test"]:
            datasets[split] = dataset.XenoCantoSpectrograms(
                self.spectrogram_path_manager, chunk_length=self.chunk_length,
                include_noise_samples=self.include_noise_samples, split=split,
                multi_label_classification=self.multi_label_classification)

            shuffle = (split == "train")
            dataloaders[split] = DataLoader(
                datasets[split], batch_size=self.batch_size, sampler=None,
                shuffle=shuffle, num_workers=self.num_workers)

        return datasets, dataloaders

    def setup_model(self):
        if self.architecture == "resnet18":
            model = models.resnet18(pretrained=True, progress=(not self.is_pipeline_run))
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, self.num_classes)

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
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=0)
        elif self.learning_rate_scheduler == "step_lr":
            scheduler = torch.optim.StepLR(optimizer, step_size=self.learning_rate_scheduler_step_size,
                                           gamma=self.learning_rate_scheduler_gamma)
        else:
            scheduler = None
        return loss, optimizer, scheduler

    def train_model(self):
        model = self.setup_model()
        device = ModelTrainer.setup_device()
        loss_function, optimizer, scheduler = self.setup_optimization(model)

        model_tracker = tracker.ModelTracker(self.spectrogram_path_manager, self.experiment_name, self.datasets["train"].id_to_class_mapping(), self.is_pipeline_run, model, self.multi_label_classification)

        model.to(device)

        since = time.time()

        logger.info("Number of species: %i", self.num_classes)

        for epoch in range(self.num_epochs):
            logger.info("Epoch %i/%i", epoch + 1, self.num_epochs)
            logger.info("----------")

            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                model_metrics = metrics.Metrics(num_classes=self.num_classes,
                                                multi_label=self.multi_label_classification)

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

        time_elapsed = time.time() - since
        logger.info("Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60))

        model_tracker.save_best_models(self.logger.get_run_id())

        self.logger.print_model_summary(model_tracker.best_average_epoch,
                                        model_tracker.best_average_metrics,
                                        model_tracker.best_minimum_epoch,
                                        model_tracker.best_minimum_metrics,
                                        model_tracker.best_epochs_per_class if self.multi_label_classification else None,
                                        model_tracker.best_metrics_per_class if self.multi_label_classification else None)

        if self.is_pipeline_run:
            self.logger.log_metrics_in_kubeflow(model_tracker.best_average_metrics,
                                                model_tracker.best_minimum_metrics,
                                                model_tracker.best_metrics_per_class if self.multi_label_classification else None)

        self.logger.finish()

        model.load_state_dict(model_tracker.best_average_model)
        best_average_model = model
        model.load_state_dict(model_tracker.best_minimum_model)
        best_minimum_model = model
        best_models_per_class = {}
        for class_name, state_dict in model_tracker.best_models_per_class.items():
            model.load_state_dict(state_dict)
            best_models_per_class[class_name] = model

        return best_average_model, best_minimum_model, best_models_per_class
