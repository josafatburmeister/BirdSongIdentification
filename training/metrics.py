import torch


class Metrics:
    def __init__(self, num_classes: int, multi_label: bool = False):
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.reset()

    def reset(self):
        self.tp = torch.zeros(self.num_classes)
        self.fp = torch.zeros(self.num_classes)
        self.fn = torch.zeros(self.num_classes)
        self.tn = torch.zeros(self.num_classes)
        self.loss = 0.0

    def update(self, predictions: torch.tensor, labels: torch.tensor, loss: torch.tensor):
        self.loss += loss.item() * predictions.shape[0]
        for prediction, label in zip(predictions, labels):
            if self.multi_label:
                correct_predictions = prediction.eq(label).int()
                incorrect_predictions = torch.ones(self.num_classes) - prediction.eq(label).int()

                self.tp += correct_predictions * label
                self.tn += correct_predictions * (torch.ones(self.num_classes) - label)
                self.fp += incorrect_predictions * (torch.ones(self.num_classes) - label)
                self.fn += incorrect_predictions * label
            else:
                if prediction == label:
                    self.tp += torch.zeros(self.num_classes).scatter_(0, prediction, 1)
                    self.tn += torch.ones(self.num_classes).scatter_(0, prediction, 0)
                else:
                    self.tn += torch.ones(self.num_classes).scatter_(0, torch.tensor([prediction, label]), 0)
                    self.fp += torch.zeros(self.num_classes).scatter_(0, prediction, 1)
                    self.fn += torch.zeros(self.num_classes).scatter_(0, label, 1)

    def precision(self):
        return self.tp / (self.tp + self.fp)

    def recall(self):
        return self.tp / (self.tp + self.fn)

    def f1_score(self):
        return 2 * (self.precision() * self.recall()) / (self.precision() + self.recall())

    def accuracy(self):
        return ((self.tp).sum() / (self.tp + self.fp + self.tn + self.fn)[0]).item()

    def average_loss(self):
        return self.loss / (self.tp + self.fp + self.tn + self.fn)[0].item()
