from .base import TaskAdapter
from .classification import ClassificationTaskAdapter
from .detection import DetectionTaskAdapter, YOLOXPerObjectProvider

__all__ = ['TaskAdapter', 'ClassificationTaskAdapter', 'DetectionTaskAdapter', 'YOLOXPerObjectProvider']
