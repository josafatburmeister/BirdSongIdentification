import torch
from torch import Tensor


class Metrics:
    def __init__(self, num_classes: int, multi_label: bool = False, device: torch.device = torch.device('cpu')):
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.device = device
        self.tp = torch.zeros(self.num_classes).to(device)
        self.fp = torch.zeros(self.num_classes).to(device)
        self.fn = torch.zeros(self.num_classes).to(device)
        self.tn = torch.zeros(self.num_classes).to(device)
        self.loss = 0.0

    def reset(self):
        self.tp = torch.zeros(self.num_classes).to(self.device)
        self.fp = torch.zeros(self.num_classes).to(self.device)
        self.fn = torch.zeros(self.num_classes).to(self.device)
        self.tn = torch.zeros(self.num_classes).to(self.device)
        self.loss = 0.0

    def update(self, predictions: torch.tensor, labels: torch.tensor, loss: torch.tensor = None):
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
                    self.tn += torch.ones(self.num_classes).to(self.device)\
                        .scatter_(0, torch.tensor([prediction, label]).to(self.device), 0)
                    self.fp += torch.zeros(self.num_classes).to(self.device).scatter_(0, prediction, 1)
                    self.fn += torch.zeros(self.num_classes).to(self.device).scatter_(0, label, 1)

    def precision(self) -> Tensor:
        precision = self.tp / (self.tp + self.fp)
        return torch.nan_to_num(precision)

    def recall(self) -> Tensor:
        recall = self.tp / (self.tp + self.fn)
        return torch.nan_to_num(recall)

    def f1_score(self) -> Tensor:
        f1_score = 2 * (self.precision() * self.recall()) / (self.precision() + self.recall())
        return torch.nan_to_num(f1_score)

    def accuracy(self) -> float:
        return (self.tp.sum() / (self.tp + self.fp + self.tn + self.fn)[0]).item()

    def average_loss(self) -> float:
        return self.loss / (self.tp + self.fp + self.tn + self.fn)[0].item()
