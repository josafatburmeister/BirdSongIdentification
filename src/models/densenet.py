from typing import Iterator, List, Optional, Union

import torch.nn.functional as F
from torch import flatten, nn, Tensor
from torch.nn import Parameter
from torchvision.models import densenet
from verboselogs import VerboseLogger


class DenseNet121TransferLearning(densenet.DenseNet):
    """
    A customized version of Torchvision's DenseNet121 implementation, which allows fine-tuning of certain layers of a
    DenseNet121 pre-trained on the ImageNet dataset.
    """

    def __init__(self, num_classes: int = 2, layers_to_unfreeze: Optional[Union[str, List[str]]] = None,
                 logger: Optional[VerboseLogger] = None, p_dropout: float = 0) -> None:
        """

        Args:
            num_classes: Number of classes in the dataset used to fine-tune the model.
            layers_to_unfreeze: List of model layer names to be unfrozen for fine-tuning; if set to "all", all model
                layers will be fine-tuned.
            logger: Logger object used to print status information during model setup.
            p_dropout: Probability of dropout before the fully-connected layer.
        """

        # initialize DenseNet121 with 1000 classes to allow weight loading from pretrained model, in_features=1024
        # arch, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, pretrained, progress
        super().__init__(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4,
                         num_classes=1000)

        self.dropout = nn.Dropout(p=p_dropout)

        # load weights of pretrained model
        densenet._load_state_dict(self, densenet.model_urls["densenet121"], progress=True)

        fc_input_size = self.classifier.in_features
        self.classifier = nn.Linear(fc_input_size, num_classes)

        if not layers_to_unfreeze:
            # unfreeze these model layers per default
            layers_to_unfreeze = ["denseblock3", "transition3", "denseblock4", "norm5", "classifier"]

        # unfreeze the selected model layers for fine-tuning
        for outer_name, outer_child in self.named_children():
            if outer_name == "features":
                for inner_name, inner_child in outer_child.named_children():
                    if layers_to_unfreeze == "all" or "all" in layers_to_unfreeze or inner_name in layers_to_unfreeze:
                        if logger:
                            logger.info(f"* {inner_name} has been unfrozen.")
                    else:
                        for _, params in inner_child.named_parameters():
                            # freeze layer
                            params.requires_grad = False
            elif layers_to_unfreeze == "all" or "all" in layers_to_unfreeze or outer_name in layers_to_unfreeze:
                if logger:
                    logger.info(f"* {outer_name} has been unfrozen.")
            else:
                for _, params in outer_child.named_parameters():
                    # freeze layer
                    params.requires_grad = False

    def forward(self, x) -> Tensor:
        # the forward method is overwritten here to apply dropout before the fully-connected layer
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = flatten(out, 1)
        if self.training:
            out = self.dropout(out)
        out = self.classifier(out)
        return out

    def parameters(self, recurse=True) -> Iterator[Parameter]:
        """
        Args:
            recurse: If True, then yields parameters of this model and all submodules.

        Returns:
            An iterator over those model parameters that were unfrozen for fine-tuning.
        """

        params = super().parameters(recurse=recurse)
        # only return the parameters that should be re-trained
        return iter([parameter for parameter in params if parameter.requires_grad])
