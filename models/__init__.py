from .densenet import DenseNet121TransferLearning
from .resnet import ResnetTransferLearning

model_architectures = {"resnet18": ResnetTransferLearning, "resnet34": ResnetTransferLearning,
                       "resnet50": ResnetTransferLearning, "densenet121": DenseNet121TransferLearning}
