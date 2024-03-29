import copy
from typing import List

from tabulate import tabulate

from general import logger, FileManager
from training.training import ModelTrainer


class HyperparameterTuner:
    """
    Runs hyperparameter tuning.
    """

    @staticmethod
    def tunable_parameters() -> List[str]:
        """

        Returns:
            List of tunable hyperparameters.
        """

        return ["batch_size", "include_noise_samples", "layers_to_unfreeze", "learning_rate", "momentum", "p_dropout",
                "weight_decay"]

    def __init__(self,
                 spectrogram_file_manager: FileManager,
                 experiment_name: str,
                 **kwargs) -> None:
        """

        Args:
            spectrogram_file_manager: FileManager object that manages the directory containing the spectrograms file and
                their labels.
            experiment_name: Descriptive name of the training run / experiment.
        """

        self.experiment_name = experiment_name
        self.spectrogram_file_manager = spectrogram_file_manager
        self.kwargs = kwargs
        self.tuned_parameters = []

        self.metrics = []
        self.best_average_f1_score = 0.0
        self.best_parameters = None

    def __tune(self, parameters: dict, experiment_name: str) -> None:
        """
        Recursively combines the values of all hyperparameters to be tuned and starts corresponding training runs.

        Args:
            parameters: Model training parameters.
            experiment_name: Descriptive name of the training run / experiment.

        Returns:
            None
        """

        unresolved_parameters = False
        for hyperparameter in HyperparameterTuner.tunable_parameters():
            if hyperparameter in parameters and \
                    (hyperparameter != "layers_to_unfreeze"
                     and type(parameters[hyperparameter]) == list
                     or hyperparameter == "layers_to_unfreeze"
                     and type(parameters[hyperparameter]) == list
                     and type(parameters[hyperparameter][0]) == list):
                if hyperparameter not in self.tuned_parameters:
                    self.tuned_parameters.append(hyperparameter)
                unresolved_parameters = True
                for parameter_value in parameters[hyperparameter]:
                    new_params = copy.deepcopy(parameters)
                    new_params[hyperparameter] = parameter_value
                    name = experiment_name + "_" + hyperparameter + "_" + str(parameter_value)
                    self.__tune(new_params, name)
                break

        if not unresolved_parameters:
            logger.info("-" * 25)
            for hyperparameter in self.tuned_parameters:
                logger.info(f"{hyperparameter} = {parameters[hyperparameter]}")
            logger.info("\n")
            model_trainer = ModelTrainer(self.spectrogram_file_manager, experiment_name=experiment_name,
                                         is_hyperparameter_tuning=True, **parameters)
            best_average_f1_score = model_trainer.train_model()
            if len(self.tuned_parameters) <= 2:
                if len(self.metrics) == 0 or len(self.metrics[-1]) == len(self.kwargs[self.tuned_parameters[0]]):
                    self.metrics.append([best_average_f1_score])
                else:
                    self.metrics[-1].append(best_average_f1_score)
            if best_average_f1_score > self.best_average_f1_score:
                self.best_average_f1_score = best_average_f1_score
                self.best_parameters = parameters.copy()

    def __log_summary(self) -> None:
        """
        Logs results of hyperparameter tuning.

        Returns:
            None
        """

        logger.info("\n")
        logger.info("Best hyperparamater combination:")
        for hyperparameter in self.tuned_parameters:
            logger.info(f"{hyperparameter} = {self.best_parameters[hyperparameter]}")
        logger.info("\n")

        if len(self.tuned_parameters) <= 2:
            if len(self.tuned_parameters) == 1:
                table_headers = ["Metric"]
            else:
                table_headers = [f"{self.tuned_parameters[1]} / {self.tuned_parameters[0]}"]
            table_headers.extend(self.kwargs[self.tuned_parameters[0]])
            metrics = []
            for idx, row in enumerate(self.metrics):
                if len(self.tuned_parameters) == 1:
                    metrics_row = ["F1-score"]
                else:
                    metrics_row = [self.kwargs[self.tuned_parameters[1]][idx]]
                metrics_row.extend(row)
                metrics.append(metrics_row)
                logger.info(
                    tabulate(metrics, headers=table_headers, tablefmt='github', floatfmt=".4f", numalign="center"))

    def tune_model(self) -> None:
        """
        Runs hyperparameter tuning.

        Returns:
            None
        """

        logger.info("Hyperparameter Tuning \n")
        self.__tune(self.kwargs.copy(), self.experiment_name)
        self.__log_summary()
