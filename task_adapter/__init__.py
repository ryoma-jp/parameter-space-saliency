from .base import TaskAdapter
from .classification import ClassificationTaskAdapter
from .detection import DetectionTaskAdapter

__all__ = ['TaskAdapter', 'ClassificationTaskAdapter', 'DetectionTaskAdapter']
