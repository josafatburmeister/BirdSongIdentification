from torchvision.models import resnet
from torch.hub import load_state_dict_from_url
from torch import nn


class ResnetTransferLearning(resnet.ResNet):
    def __init__(self, architecture: str = "resnet18", num_classes: int = 2, layers_to_unfreeze=None, logger=None):
        # initialize Resnet with 1000 classes to allow weight loading from pretrained model, in_features=512
        if architecture == "resnet18":
            super().__init__(resnet.BasicBlock, layers=[2, 2, 2, 2], num_classes=1000)
        elif architecture == "resnet34":
            super().__init__(resnet.BasicBlock, layers=[3, 4, 6, 3], num_classes=1000)
        elif architecture == "resnet50":
            super().__init__(resnet.Bottleneck, layers=[3, 4, 6, 3], num_classes=1000)
        else:
            raise NameError("Invalid architecture")

        # load weights of pretrained model
        state_dict = load_state_dict_from_url(
            resnet.model_urls[architecture], progress=True)
        self.load_state_dict(state_dict)

        fc_input_size = self.fc.in_features
        self.fc = nn.Linear(fc_input_size, num_classes)  # fc_input_size = 512

        if layers_to_unfreeze is None:
            layers_to_unfreeze = ["layer3", "layer4", "avgpool", "fc"]

        # unfreeze the selected layers for fine-tuning
        for name, child in self.named_children():
            if layers_to_unfreeze == "all" or name in layers_to_unfreeze:
                if logger:
                    logger.info(f"* {name} has been unfrozen.")
            else:
                for _, params in child.named_parameters():
                    # freeze layer
                    params.requires_grad = False

    def parameters(self):
        params = super().parameters()
        # only return the paramaters that should be re-trained
        return iter([parameter for parameter in params if parameter.requires_grad])

    def load_state_dict(self, state_dict):
        self.load_state_dict(state_dict)
