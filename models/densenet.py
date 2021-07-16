from torchvision.models import densenet
from torch import flatten, nn, save, load
import torch.nn.functional as F

class DenseNet121TransferLearning(densenet.DenseNet):
    def __init__(self, num_classes=2, layers_to_unfreeze=None, logger=None, p_dropout=0):
        # initialize Densest with 1000 classes to allow weight loading from pretrained model, in_features=1024
        # arch, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, pretrained, progress
        super().__init__(growth_rate=32, block_config=(6, 12, 24, 16),
                         num_init_features=64, bn_size=4, num_classes=1000)

        self.dropout = nn.Dropout(p=p_dropout)

        # load weights of pretrained model
        densenet._load_state_dict(self, densenet.model_urls["densenet121"], progress=True)

        fc_input_size = self.classifier.in_features
        self.classifier = nn.Linear(fc_input_size, num_classes)  # fc_input_size=1024

        if layers_to_unfreeze is None:
            layers_to_unfreeze = ["denseblock3", "transition3", "denseblock4", "norm5", "classifier"]

        # unfreeze the selected layers for fine-tuning
        for name, child in self.named_children():
            if name == "features":
                for name, child in child.named_children():
                    if layers_to_unfreeze == "all" or name in layers_to_unfreeze:
                        if logger:
                            logger.info(f"* {name} has been unfrozen.")
                    else:
                        for _, params in child.named_parameters():
                            # freeze layer
                            params.requires_grad = False
            elif layers_to_unfreeze == "all" or name in layers_to_unfreeze:
                if logger:
                    logger.info(f"* {name} has been unfrozen.")
            else:
                for _, params in child.named_parameters():
                    # freeze layer
                    params.requires_grad = False

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = flatten(out, 1)
        if self.training:
            out = self.dropout(out)
        out = self.classifier(out)
        return out

    def parameters(self):
        params = super().parameters()
        # only return the parameters that should be re-trained
        return iter([parameter for parameter in params if parameter.requires_grad])

    @classmethod
    def load_model(cls, path: str):
        """
        stores the model for later retrieval
        :param path: file from which to store the model
        """
        model = cls()
        model.load_state_dict(load(path)['state_dict'])
        return model
