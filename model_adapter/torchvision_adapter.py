import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from .base import ModelAdapter

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


class TorchvisionAdapter(ModelAdapter):
    """Adapter for torchvision pretrained classification models.

    Any model available via ``torchvision.models.<name>`` can be used;
    the existing four (resnet50, vgg19, densenet121, inception_v3) are
    supported out of the box, as is any future torchvision model.
    """

    def __init__(self, model_name: str, pretrained: bool = True):
        self.model_name = model_name
        self.pretrained = pretrained

    def build_model(self) -> nn.Module:
        constructor = getattr(torchvision.models, self.model_name, None)
        if constructor is None:
            raise ValueError(
                f"torchvision.models has no model named '{self.model_name}'. "
                "Check spelling or use source='custom_module'."
            )
        return constructor(pretrained=self.pretrained)

    def get_preprocess(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])

    def get_inv_preprocess(self) -> transforms.Compose:
        inv_std  = tuple(1.0 / s for s in _IMAGENET_STD)
        inv_mean = tuple(-m         for m in _IMAGENET_MEAN)
        return transforms.Compose([
            transforms.Normalize(mean=(0., 0., 0.), std=inv_std),
            transforms.Normalize(mean=inv_mean,    std=(1., 1., 1.)),
        ])
