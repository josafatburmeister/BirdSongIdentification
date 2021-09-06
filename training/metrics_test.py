import unittest

import torch

from training import metrics


class TestMetricsSingleLabel(unittest.TestCase):

    def setUp(self):
        self.metrics = metrics.Metrics(3)
        self.metrics.reset()

    def test_single_label_metrics_all_correct(self):
        predictions = torch.tensor([1, 0, 2])
        labels = torch.tensor([1, 0, 2])
        self.metrics.update(predictions, labels)
        self.assertTrue(torch.all(self.metrics.tp.eq(torch.tensor([1, 1, 1]))),
                        "Computes true positives correctly if all predictions are correct")
        self.assertTrue(torch.all(self.metrics.tn.eq(torch.tensor([2, 2, 2]))),
                        "Computes true negatives correctly if all predictions are correct")
        self.assertTrue(torch.all(self.metrics.fp.eq(torch.tensor([0, 0, 0]))),
                        "Computes false positives correctly if all predictions are correct")
        self.assertTrue(torch.all(self.metrics.fn.eq(torch.tensor([0, 0, 0]))),
                        "Computes false negatives correctly if all predictions are correct")

    def test_single_label_metrics_all_incorrect(self):
        predictions = torch.tensor([1, 0, 2])
        labels = torch.tensor([2, 1, 1])
        self.metrics.update(predictions, labels)
        self.assertTrue(torch.all(self.metrics.tp.eq(torch.tensor([0, 0, 0]))),
                        "Computes true positives correctly if all predictions are incorrect")
        self.assertTrue(torch.all(self.metrics.tn.eq(torch.tensor([2, 0, 1]))),
                        "Computes true negatives correctly if all predictions are incorrect")
        self.assertTrue(torch.all(self.metrics.fp.eq(torch.tensor([1, 1, 1]))),
                        "Computes false positives correctly if all predictions are incorrect")
        self.assertTrue(torch.all(self.metrics.fn.eq(torch.tensor([0, 2, 1]))),
                        "Computes false negatives correctly if all predictions are incorrect")

    def test_single_label_metrics_some_incorrect(self):
        predictions = torch.tensor([1, 0, 2])
        labels = torch.tensor([1, 1, 1])
        self.metrics.update(predictions, labels)
        self.assertTrue(torch.all(self.metrics.tp.eq(torch.tensor([0, 1, 0]))),
                        "Computes true positives correctly if some predictions are incorrect")
        self.assertTrue(torch.all(self.metrics.tn.eq(torch.tensor([2, 0, 2]))),
                        "Computes true negatives correctly if some predictions are incorrect")
        self.assertTrue(torch.all(self.metrics.fp.eq(torch.tensor([1, 0, 1]))),
                        "Computes false positives correctly if some predictions are incorrect")
        self.assertTrue(torch.all(self.metrics.fn.eq(torch.tensor([0, 2, 0]))),
                        "Computes false negatives correctly if some predictions are incorrect")

    def test_precision_all_correct(self):
        predictions = torch.tensor([1, 0, 2])
        labels = torch.tensor([1, 0, 2])
        self.metrics.update(predictions, labels)
        self.assertTrue(torch.all(self.metrics.precision().eq(torch.tensor([1, 1, 1]))),
                        "Computes precision correctly if all predictions are correct")

    def test_precision_all_incorrect(self):
        predictions = torch.tensor([1, 0, 2])
        labels = torch.tensor([2, 1, 1])
        self.metrics.update(predictions, labels)
        self.assertTrue(torch.all(self.metrics.precision().eq(torch.tensor([0, 0, 0]))),
                        "Computes precision correctly if all predictions are incorrect")

    def test_precision_some_incorrect(self):
        predictions = torch.tensor([1, 1, 2])
        labels = torch.tensor([1, 0, 2])
        self.metrics.update(predictions, labels)
        self.assertTrue(torch.allclose(self.metrics.precision(), torch.tensor([float('nan'), 0.5, 1]), equal_nan=True),
                        "Computes precision correctly if some predictions are incorrect")

    def test_recall_all_correct(self):
        predictions = torch.tensor([1, 0, 2])
        labels = torch.tensor([1, 0, 2])
        self.metrics.update(predictions, labels)
        self.assertTrue(torch.all(self.metrics.recall().eq(torch.tensor([1, 1, 1]))),
                        "Computes recall correctly if all predictions are correct")

    def test_recall_all_incorrect(self):
        predictions = torch.tensor([1, 0, 2])
        labels = torch.tensor([2, 1, 1])
        self.metrics.update(predictions, labels)
        self.assertTrue(torch.allclose(self.metrics.recall(), torch.tensor([float('nan'), 0, 0]), equal_nan=True),
                        "Computes recall correctly if all predictions are incorrect")

    def test_recall_some_incorrect(self):
        predictions = torch.tensor([1, 1, 2])
        labels = torch.tensor([1, 0, 2])
        self.metrics.update(predictions, labels)
        self.assertTrue(torch.all(self.metrics.recall().eq(torch.tensor([0, 1, 1]))),
                        "Computes recall correctly if some predictions are incorrect")


class TestMetricsMultiLabel(unittest.TestCase):

    def setUp(self):
        self.metrics = metrics.Metrics(3, multi_label=True)
        self.metrics.reset()

    def test_multi_label_metrics_all_correct(self):
        predictions = torch.tensor([[0, 1, 1], [1, 0, 0], [1, 1, 1]])
        labels = torch.tensor([[0, 1, 1], [1, 0, 0], [1, 1, 1]])
        self.metrics.update(predictions, labels)
        self.assertTrue(torch.all(self.metrics.tp.eq(torch.tensor([2, 2, 2]))),
                        "Computes true positives correctly if all predictions are correct")
        self.assertTrue(torch.all(self.metrics.tn.eq(torch.tensor([1, 1, 1]))),
                        "Computes true negatives correctly if all predictions are correct")
        self.assertTrue(torch.all(self.metrics.fp.eq(torch.tensor([0, 0, 0]))),
                        "Computes false positives correctly if all predictions are correct")
        self.assertTrue(torch.all(self.metrics.fn.eq(torch.tensor([0, 0, 0]))),
                        "Computes false negatives correctly if all predictions are correct")

    def test_single_label_metrics_all_incorrect(self):
        predictions = torch.tensor([[0, 1, 1], [1, 0, 0], [0, 0, 1]])
        labels = torch.tensor([[1, 0, 0], [0, 1, 1], [1, 1, 0]])
        self.metrics.update(predictions, labels)

        self.assertTrue(torch.all(self.metrics.tp.eq(torch.tensor([0, 0, 0]))),
                        "Computes true positives correctly if all predictions are incorrect")
        self.assertTrue(torch.all(self.metrics.tn.eq(torch.tensor([0, 0, 0]))),
                        "Computes true negatives correctly if all predictions are incorrect")
        self.assertTrue(torch.all(self.metrics.fp.eq(torch.tensor([1, 1, 2]))),
                        "Computes false positives correctly if all predictions are incorrect")
        self.assertTrue(torch.all(self.metrics.fn.eq(torch.tensor([2, 2, 1]))),
                        "Computes false negatives correctly if all predictions are incorrect")

    def test_single_label_metrics_some_incorrect(self):
        predictions = torch.tensor([[1, 0, 1], [1, 0, 0], [1, 1, 0]])
        labels = torch.tensor([[0, 0, 1], [0, 0, 0], [0, 1, 1]])
        self.metrics.update(predictions, labels)

        self.assertTrue(torch.all(self.metrics.tp.eq(torch.tensor([0, 1, 1]))),
                        "Computes true positives correctly if some predictions are incorrect")
        self.assertTrue(torch.all(self.metrics.tn.eq(torch.tensor([0, 2, 1]))),
                        "Computes true negatives correctly if some predictions are incorrect")
        self.assertTrue(torch.all(self.metrics.fp.eq(torch.tensor([3, 0, 0]))),
                        "Computes false positives correctly if some predictions are incorrect")
        self.assertTrue(torch.all(self.metrics.fn.eq(torch.tensor([0, 0, 1]))),
                        "Computes false negatives correctly if some predictions are incorrect")


if __name__ == '__main__':
    unittest.main()
