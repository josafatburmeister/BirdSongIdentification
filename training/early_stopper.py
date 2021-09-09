from training.metrics import Metrics


class EarlyStopper:
    """
    Monitors model training and stops it when model performance no longer improves.
    """

    def __init__(self, monitor: str, patience: int, min_change: float) -> None:
        """

        Args:
            monitor: Performance metric that should be monitored. Either "f1-score", "accuracy" or "average_loss".
            patience: Number of epochs without performance increase that waited before stopping the model training.
            min_change: Minimum performance increase per epoch that justifies continuing model training.
        """

        if monitor in {'f1-score', 'accuracy', 'average_loss'}:
            self.monitor = monitor
        else:
            raise ValueError('Monitor has to be in {\'f1-score\', \'accuracy\', \'average_loss\'}. Monitor given was: ',
                             monitor)
        self.patience = patience
        self.min_change = min_change
        self.waitCounter = 0
        self.best_value = -1e15

    def check_early_stopping(self, model_metrics: Metrics) -> bool:
        """
        Checks if the model training should be stopped after the current epoch.

        Args:
            model_metrics: Metrics object representing the model performance in the current epoch.

        Returns:
            Whether model training should be stopped or not.
        """

        if not self.monitor:
            raise Exception('Early Stopping has not been setup.')
        if self.monitor == 'f1-score':
            current_value = model_metrics.f1_score().mean().item()
        elif self.monitor == 'accuracy':
            current_value = model_metrics.accuracy()
        elif self.monitor == 'average_loss':
            current_value = -model_metrics.average_loss()
        else:
            raise ValueError('monitor has invalid Value: ', self.monitor)

        if current_value is None:
            pass
        else:
            if (current_value - self.best_value) > self.min_change:
                self.best_value = current_value
                self.waitCounter = 1
            else:
                if self.waitCounter >= self.patience:
                    return True
                self.waitCounter += 1
        return False
