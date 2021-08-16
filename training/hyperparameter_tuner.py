from training.training import ModelTrainer
from general.logging import logger
from tabulate import tabulate
from typing import List


class HyperparameterTuner:
    @staticmethod
    def tunable_parameters():
        return ["batch_size", "learning_rate", "momentum", "weight_decay"]

    def __init__(self,
                 spectrogram_path_manager,
                 experiment_name: str,
                 **kwargs):
        self.experiment_name = experiment_name
        self.spectrogram_path_manager = spectrogram_path_manager
        self.kwargs = kwargs
        self.tuned_parameters = []

        self.metrics = []
        self.best_average_f1_score = 0.0
        self.best_parameters = None

    def tune(self, parameters: dict, experiment_name):
        unresolved_parameters = False
        for hyperparameter in HyperparameterTuner.tunable_parameters():
            if hyperparameter in parameters and type(parameters[hyperparameter]) == list:
                self.tuned_parameters.append(hyperparameter)
                unresolved_parameters = True
                for parameter_value in parameters[hyperparameter]:
                    new_params = parameters
                    new_params[hyperparameter] = parameter_value
                    experiment_name = experiment_name + hyperparameter + str(parameter_value)
                    self.tune(new_params, experiment_name)

        if not unresolved_parameters:
            logger.info("-" * 25)
            for hyperparameter in self.tuned_parameters:
                logger.info(f"{hyperparameter} = {parameters[hyperparameter]}")
            logger.info("\n")
            experiment_name = experiment_name
            model_trainer = ModelTrainer(self.spectrogram_path_manager, experiment_name=experiment_name,
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

    def log_summary(self):
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

    def tune_model(self):
        logger.info("Hyperparameter Tuning \n")
        self.tune(self.kwargs.copy(), self.experiment_name)
        self.log_summary()

