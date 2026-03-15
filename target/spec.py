from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TargetType(str, Enum):
    TRUE_LABEL = 'true_label'
    PREDICTED_TOP1 = 'predicted_top1'
    SPECIFIED_CLASS = 'specified_class'


@dataclass
class TargetSpec:
    """Specification of what to compute saliency with respect to.

    Attributes:
        type: One of true_label, predicted_top1, specified_class.
        class_id: Required when type is 'specified_class'.
    """

    type: TargetType = TargetType.TRUE_LABEL
    class_id: Optional[int] = None

    def __post_init__(self):
        if self.type == TargetType.SPECIFIED_CLASS and self.class_id is None:
            raise ValueError("class_id must be set when type is 'specified_class'")

    @classmethod
    def from_args(cls, target_type: str, target_class_id: Optional[int] = None) -> 'TargetSpec':
        return cls(type=TargetType(target_type), class_id=target_class_id)
