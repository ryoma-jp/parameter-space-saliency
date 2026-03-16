import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn


class YOLOXTinyWrapper(nn.Module):
    """Thin nn.Module wrapper that builds YOLOX Tiny from vendored YOLOX code.

    This class is designed to be instantiated via CustomModuleAdapter using
    --model_class_path model_adapter.yolox_tiny_wrapper.YOLOXTinyWrapper
    """

    def __init__(self):
        super().__init__()
        self._ensure_yolox_on_path()

        from yolox.exp import get_exp

        exp = get_exp(exp_file=self._exp_file_path())
        exp.num_classes = 80
        self.model = exp.get_model()

    @staticmethod
    def _repo_root() -> str:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @classmethod
    def _yolox_root(cls) -> str:
        return os.path.join(cls._repo_root(), "externals", "YOLOX")

    @classmethod
    def _exp_file_path(cls) -> str:
        return os.path.join(cls._yolox_root(), "exps", "default", "yolox_tiny.py")

    @classmethod
    def _ensure_yolox_on_path(cls) -> None:
        yolox_root = cls._yolox_root()
        if yolox_root not in sys.path:
            sys.path.insert(0, yolox_root)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOLOX preproc typically feeds [0, 255] float images.
        if x.dtype.is_floating_point and x.detach().max().item() <= 1.5:
            x = x * 255.0
        return self.model(x)

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load either wrapped keys ('model.xxx') or plain YOLOX keys ('xxx')."""
        if not state_dict:
            return self.model.load_state_dict(state_dict, strict=strict)

        first_key = next(iter(state_dict.keys()))
        if first_key.startswith("model."):
            stripped = OrderedDict((k[len("model."):], v) for k, v in state_dict.items())
            return self.model.load_state_dict(stripped, strict=strict)
        return self.model.load_state_dict(state_dict, strict=strict)
