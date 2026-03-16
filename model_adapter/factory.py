from typing import Any, Dict

from .base import ModelAdapter
from .torchvision_adapter import TorchvisionAdapter
from .custom_module_adapter import CustomModuleAdapter


def build_model_adapter(spec: Dict[str, Any]) -> ModelAdapter:
    """Build a :class:`ModelAdapter` from a specification dict.

    Supported sources
    -----------------
    ``'torchvision'``
        Load a pretrained ``torchvision.models`` model by name.
        Required keys: ``name``.
        Optional keys: ``pretrained`` (default ``True``).

    ``'custom_module'``
        Load a user-defined ``nn.Module`` from a fully-qualified class path or
        factory callable path.
        Required keys: ``class_path``.
        Optional keys: ``weights_path``, ``preprocess``, ``model_kwargs``,
        ``import_roots``, ``state_dict_target_path``.

    Examples::

        # torchvision
        build_model_adapter({'source': 'torchvision', 'name': 'resnet50'})

        # user-defined model
        build_model_adapter({
            'source': 'custom_module',
            'class_path': 'mypkg.models.MyNet',
            'weights_path': 'checkpoints/mynet.pth',
            'preprocess': {
                'resize': 256, 'crop': 224,
                'mean': [0.5, 0.5, 0.5],
                'std':  [0.5, 0.5, 0.5],
            },
        })
    """
    source = spec.get('source', 'torchvision')

    if source == 'torchvision':
        name = spec.get('name')
        if not name:
            raise ValueError("'name' is required for source='torchvision'")
        return TorchvisionAdapter(
            model_name=name,
            pretrained=spec.get('pretrained', True),
        )

    if source == 'custom_module':
        class_path = spec.get('class_path')
        if not class_path:
            raise ValueError("'class_path' is required for source='custom_module'")
        return CustomModuleAdapter(
            class_path=class_path,
            weights_path=spec.get('weights_path'),
            preprocess_cfg=spec.get('preprocess'),
            model_kwargs=spec.get('model_kwargs'),
            import_roots=spec.get('import_roots'),
            state_dict_target_path=spec.get('state_dict_target_path'),
        )

    raise ValueError(
        f"Unknown model source: '{source}'. "
        "Choose 'torchvision' or 'custom_module'."
    )
