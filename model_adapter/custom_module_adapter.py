import importlib
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from .base import ModelAdapter


class _LetterboxTransform:
    """PIL Image -> CHW float32 tensor via letterbox resize, matching YOLOX ``preproc()``.

    The output tensor is in the [0, 255] range with the channel order specified
    by *channel_order* (``'bgr'`` by default, matching YOLOX inference).
    """

    def __init__(self, input_size, pad_value: int = 114, channel_order: str = 'bgr'):
        # input_size: (H, W) or [H, W]
        self.input_size   = (int(input_size[0]), int(input_size[1]))
        self.pad_value    = pad_value
        self.channel_order = channel_order.lower()

    def __call__(self, pil_img):
        # PIL gives RGB HWC uint8
        img = np.array(pil_img, dtype=np.uint8)
        if self.channel_order == 'bgr':
            img = img[:, :, ::-1]  # RGB -> BGR

        src_h, src_w = img.shape[:2]
        tgt_h, tgt_w = self.input_size
        r = min(tgt_h / src_h, tgt_w / src_w)
        new_h, new_w = int(src_h * r), int(src_w * r)

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded = np.ones((tgt_h, tgt_w, 3), dtype=np.uint8) * self.pad_value
        padded[:new_h, :new_w] = resized

        # HWC -> CHW float32 [0, 255]
        tensor = torch.from_numpy(
            np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)
        )
        return tensor


class _LetterboxInvTransform:
    """Inverse of ``_LetterboxTransform`` for visualization.

    Converts CHW float32 [0, 255] (BGR or RGB) back to CHW float32 [0, 1] RGB.
    """

    def __init__(self, channel_order: str = 'bgr'):
        self.channel_order = channel_order.lower()

    def __call__(self, tensor):
        tensor = tensor / 255.0
        if self.channel_order == 'bgr':
            tensor = tensor[[2, 1, 0]]  # BGR -> RGB
        return tensor.clamp(0.0, 1.0)


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
        import_roots: Optional[list] = None,
        state_dict_target_path: Optional[str] = None,
    ):
        self.class_path     = class_path
        self.weights_path   = weights_path
        self.preprocess_cfg = preprocess_cfg or {}
        self.model_kwargs   = model_kwargs   or {}
        self.import_roots   = import_roots   or []
        self.state_dict_target_path = state_dict_target_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_class(self):
        for root in reversed(self.import_roots):
            if root and root not in sys.path:
                sys.path.insert(0, root)
        module_path, class_name = self.class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def _resolve_state_dict_target(self, model: nn.Module) -> nn.Module:
        if not self.state_dict_target_path:
            return model

        target = model
        for attr in self.state_dict_target_path.split('.'):
            target = getattr(target, attr)
        return target

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
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"Resolved object '{self.class_path}' must construct and return nn.Module, "
                f"got {type(model)!r}."
            )

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
            self._resolve_state_dict_target(model).load_state_dict(state_dict)

        return model

    def get_preprocess(self) -> transforms.Compose:
        if self.preprocess_cfg.get('letterbox', False):
            resize        = self.preprocess_cfg.get('resize', [416, 416])
            pad_value     = int(self.preprocess_cfg.get('pad_value', 114))
            channel_order = str(self.preprocess_cfg.get('channel_order', 'bgr'))
            return _LetterboxTransform(resize, pad_value, channel_order)

        resize    = self.preprocess_cfg.get('resize', 256)
        crop      = self.preprocess_cfg.get('crop',   224)
        normalize = self.preprocess_cfg.get('normalize', True)
        scale     = float(self.preprocess_cfg.get('scale', 1.0))
        mean, std = self._mean_std()

        transform_ops = [transforms.Resize(resize)]
        if crop is not None:
            transform_ops.append(transforms.CenterCrop(crop))
        transform_ops.append(transforms.ToTensor())
        if scale != 1.0:
            transform_ops.append(transforms.Lambda(lambda tensor: tensor * scale))
        if normalize:
            transform_ops.append(transforms.Normalize(mean, std))
        return transforms.Compose(transform_ops)

    def get_inv_preprocess(self) -> transforms.Compose:
        if self.preprocess_cfg.get('letterbox', False):
            channel_order = str(self.preprocess_cfg.get('channel_order', 'bgr'))
            return _LetterboxInvTransform(channel_order)

        mean, std = self._mean_std()
        normalize = self.preprocess_cfg.get('normalize', True)
        scale     = float(self.preprocess_cfg.get('scale', 1.0))
        inv_std   = tuple(1.0 / s for s in std)
        inv_mean  = tuple(-m for m in mean)

        transform_ops = []
        if normalize:
            transform_ops.extend([
                transforms.Normalize(mean=(0., 0., 0.), std=inv_std),
                transforms.Normalize(mean=inv_mean,    std=(1., 1., 1.)),
            ])
        if scale != 1.0:
            transform_ops.append(transforms.Lambda(lambda tensor: tensor / scale))
        return transforms.Compose(transform_ops)
