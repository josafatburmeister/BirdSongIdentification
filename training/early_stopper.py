from training.metrics import Metrics


class EarlyStopper:
    def __init__(self, monitor, patience, min_change) -> None:
        if monitor in {'f1_score', 'accuracy', 'average_loss'}:
            self.monitor = monitor
        else:
            raise ValueError('Monitor has to be in {\'f1_score\', \'accuracy\', \'average_loss\'}. Monitor given was: ',
                             monitor)
        self.patience = patience
        self.min_change = min_change
        self.waitCounter = 0
        self.best_value = -1e15

    def check_early_stopping(self, model_metrics: Metrics) -> bool:
        if not self.monitor:
            raise Exception('Early Stopping has not been setup.')
        if self.monitor == 'f1_score':
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
