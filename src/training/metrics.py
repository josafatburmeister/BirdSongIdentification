import torch
from torch import Tensor


class Metrics:
    """
    Computes and stores model performance metrics.
    """

    def __init__(self, num_classes: int, multi_label: bool = False, device: torch.device = torch.device('cpu')) -> None:
        """

        Args:
            num_classes: Number of classes in the dataset used to train the model.
            multi_label: Whether the model is trained as single-label classification model or as multi-label
                classification model.
            device: Pytorch device that should be used for metric calculations.
        """

        self.num_classes = num_classes
        self.multi_label = multi_label
        self.device = device
        self.tp = torch.zeros(self.num_classes).to(device)
        self.fp = torch.zeros(self.num_classes).to(device)
        self.fn = torch.zeros(self.num_classes).to(device)
        self.tn = torch.zeros(self.num_classes).to(device)
        self.loss = 0.0

    def reset(self) -> None:
        """
        Clears all metric values.

        Returns:
            None
        """

        self.tp = torch.zeros(self.num_classes).to(self.device)
        self.fp = torch.zeros(self.num_classes).to(self.device)
        self.fn = torch.zeros(self.num_classes).to(self.device)
        self.tn = torch.zeros(self.num_classes).to(self.device)
        self.loss = 0.0

    def update(self, predictions: torch.tensor, labels: torch.tensor, loss: torch.tensor = None) -> None:
        """
        Updates the aggregated metrics with the results from a training batch.

        Args:
            predictions: Model predictions for the current training batch.
            labels: Class labels for the current training batch.
            loss: Model loss on the current training batch.

        Returns:
            None.
        """

        if loss is not None:
            self.loss += loss.item() * predictions.shape[0]
        for prediction, label in zip(predictions, labels):
            prediction = prediction.to(self.device)
            label = label.to(self.device)

            if self.multi_label:
                correct_predictions = prediction.eq(label).int()
                incorrect_predictions = torch.ones(self.num_classes).to(self.device) - correct_predictions

                self.tp += correct_predictions * label
                self.tn += correct_predictions * (torch.ones(self.num_classes).to(self.device) - label)
                self.fp += incorrect_predictions * (torch.ones(self.num_classes).to(self.device) - label)
                self.fn += incorrect_predictions * label
            else:
                if prediction == label:
                    self.tp += torch.zeros(self.num_classes).to(self.device).scatter_(0, prediction, 1)
                    self.tn += torch.ones(self.num_classes).to(self.device).scatter_(0, prediction, 0)
                else:
                    self.tn += torch.ones(self.num_classes).to(self.device) \
                        .scatter_(0, torch.tensor([prediction, label]).to(self.device), 0)
                    self.fp += torch.zeros(self.num_classes).to(self.device).scatter_(0, prediction, 1)
                    self.fn += torch.zeros(self.num_classes).to(self.device).scatter_(0, label, 1)

    def precision(self) -> Tensor:
        """

        Returns:
            Class-wise precision.
        """

        precision = self.tp / (self.tp + self.fp)
        precision[torch.isnan(precision)] = 0.0
        return precision

    def recall(self) -> Tensor:
        """

        Returns:
            Class-wise recall.
        """

        recall = self.tp / (self.tp + self.fn)
        recall[torch.isnan(recall)] = 0.0
        return recall

    def f1_score(self) -> Tensor:
        """

        Returns:
            Class-wise F1-score.
        """

        f1_score = 2 * (self.precision() * self.recall()) / (self.precision() + self.recall())
        f1_score[torch.isnan(f1_score)] = 0.0
        return f1_score

    def accuracy(self) -> float:
        """

        Returns:
            Accuracy.
        """

        return (self.tp.sum() / (self.tp + self.fp + self.tn + self.fn)[0]).item()

    def average_loss(self) -> float:
        """

        Returns:
            Average model loss.
        """

        return self.loss / (self.tp + self.fp + self.tn + self.fn)[0].item()
