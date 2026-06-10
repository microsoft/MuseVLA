"""
MuseVLA Inference Module

Provides modular inference classes for different output formats:
- MuseVLAInference6DPos: 6D EEF position output (18D per hand)
- MuseVLAInference6DPosDelta: 6D EEF position delta output (18D per hand)
- MuseVLAInference7DJointAngle: 7D joint angle output (19D per hand)
"""

from .inference_6d_pos import MuseVLAInference6DPos
from .inference_6d_pos_delta import MuseVLAInference6DPosDelta
from .inference_7d_joint_angle import MuseVLAInference7DJointAngle

__all__ = [
    "MuseVLAInference6DPos",
    "MuseVLAInference6DPosDelta",
    "MuseVLAInference7DJointAngle",
]
