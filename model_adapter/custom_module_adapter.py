import importlib
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from .base import ModelAdapter


class CustomModuleAdapter(ModelAdapter):
    """Adapter for user-defined ``nn.Module`` classes.

    Args:
        class_path:     Fully-qualified class path, e.g. ``'mypkg.models.MyNet'``.
        weights_path:   Path to a ``.pth`` checkpoint.  The file may contain
                        a raw ``state_dict`` or a dict with a ``'state_dict'``
                        / ``'model'`` key.
        preprocess_cfg: Optional dict to override default preprocessing.
                        Supported keys: ``resize`` (int), ``crop`` (int),
                        ``mean`` (list[float]), ``std`` (list[float]).
                        Defaults to ImageNet statistics with 256/224 resize/crop.
        model_kwargs:   Keyword arguments forwarded to the model constructor.

    Example::

        adapter = CustomModuleAdapter(
            class_path='mypkg.models.MyResNet',
            weights_path='checkpoints/mynet.pth',
            preprocess_cfg={
                'resize': 256, 'crop': 224,
                'mean': [0.5, 0.5, 0.5],
                'std':  [0.5, 0.5, 0.5],
            },
        )
        net = adapter.build_model()
    """

    _DEFAULT_MEAN = (0.485, 0.456, 0.406)
    _DEFAULT_STD  = (0.229, 0.224, 0.225)

    def __init__(
        self,
        class_path: str,
        weights_path: Optional[str] = None,
        preprocess_cfg: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
    ):
        self.class_path     = class_path
        self.weights_path   = weights_path
        self.preprocess_cfg = preprocess_cfg or {}
        self.model_kwargs   = model_kwargs   or {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_class(self):
        module_path, class_name = self.class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def _mean_std(self) -> Tuple[tuple, tuple]:
        mean = tuple(self.preprocess_cfg.get('mean', self._DEFAULT_MEAN))
        std  = tuple(self.preprocess_cfg.get('std',  self._DEFAULT_STD))
        return mean, std

    # ------------------------------------------------------------------
    # ModelAdapter interface
    # ------------------------------------------------------------------

    def build_model(self) -> nn.Module:
        cls   = self._load_class()
        model = cls(**self.model_kwargs)

        if self.weights_path is not None:
            checkpoint = torch.load(self.weights_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                state_dict = (
                    checkpoint.get('state_dict')
                    or checkpoint.get('model')
                    or checkpoint
                )
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)

        return model

    def get_preprocess(self) -> transforms.Compose:
        resize    = self.preprocess_cfg.get('resize', 256)
        crop      = self.preprocess_cfg.get('crop',   224)
        mean, std = self._mean_std()
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    def get_inv_preprocess(self) -> transforms.Compose:
        mean, std = self._mean_std()
        inv_std   = tuple(1.0 / s for s in std)
        inv_mean  = tuple(-m for m in mean)
        return transforms.Compose([
            transforms.Normalize(mean=(0., 0., 0.), std=inv_std),
            transforms.Normalize(mean=inv_mean,    std=(1., 1., 1.)),
        ])
