"""
MuseVLA Inference with 6D Position Delta Output

Output format: 18D per hand (6D EEF pose delta + 12D hand joints)
- Input state: 18D (right hand only)
- Output action: (chunk_size, 18) right hand actions (delta format)

Note: This class is identical to MuseVLAInference6DPos in terms of dimensions.
The "delta" refers to the model training configuration, not the inference code.
"""

from .inference_6d_pos import MuseVLAInference6DPos


class MuseVLAInference6DPosDelta(MuseVLAInference6DPos):
    """
    MuseVLA inference with 6D EEF position delta output.

    This is functionally identical to MuseVLAInference6DPos.
    The distinction is in the model checkpoint used (trained with delta targets).

    State/Action format per hand: [6D EEF pose, 12D hand joints] = 18D
    Total (both hands): 36D
    """

    pass
