"""
MuseVLA Inference with 6D Position Output

Output format: 18D per hand (6D EEF pose + 12D hand joints)
- Input state: 18D (right hand only)
- Output action: (chunk_size, 18) right hand actions
"""

from typing import Optional

import numpy as np
import torch

from .base import MuseVLAInferenceBase
from musevla.datasets.dataset_utils import ActionFeature, StateFeature
from musevla.datasets.robot_dataset import (
    transfer_human_to_xhand,
    pad_state_robot,
    pad_action,
)


class MuseVLAInference6DPos(MuseVLAInferenceBase):
    """
    MuseVLA inference with 6D EEF position output.

    State/Action format per hand: [6D EEF pose, 12D hand joints] = 18D
    Total (both hands): 36D
    """

    STATE_DIM = 18  # Per hand
    ACTION_DIM = 18  # Per hand
    FULL_STATE_DIM = 36  # Both hands
    FULL_ACTION_DIM = 36  # Both hands

    def convert_input_state(self, state: np.ndarray) -> np.ndarray:
        """
        Convert client's 18D right-hand state to 36D full state.

        Client sends right hand 18D: [6D EEF pose, 12D hand joints]
        Convert to 36D: [18D left (zeros), 18D right]
        """
        state_36d = np.zeros(36, dtype=np.float32)
        state_36d[18:36] = state
        return state_36d

    def convert_output_action(self, action: np.ndarray) -> np.ndarray:
        """
        Convert model's 36D action output to 18D right-hand action.

        Model outputs 36D: [18D left, 18D right]
        Extract right hand 18D: [6D EEF pose, 12D hand joints]
        """
        return action[:, 18:36]

    def prepare_state_and_masks(
        self,
        state_full: np.ndarray,
        state_mask: Optional[np.ndarray],
        action_mask: Optional[np.ndarray],
    ) -> tuple:
        """
        Prepare state tensor and masks for VLA prediction.

        For 6D position format, no extra 1-DoF extraction is needed.
        """
        state_dim = self.FULL_STATE_DIM  # 36

        if state_mask is None:
            state_mask = np.zeros(state_dim, dtype=bool)
            state_mask[18:36] = True  # Enable right hand only

        if action_mask is None:
            action_mask = np.zeros((self.chunk_size, state_dim), dtype=bool)
            action_mask[:, 18:36] = True  # Predict right hand only

        # Normalize state
        norm_state = self.normalizer.normalize_state(state_full)

        # Convert to tensors
        norm_state_tensor = torch.from_numpy(norm_state).float()
        state_mask_tensor = torch.from_numpy(state_mask).bool()
        action_mask_tensor = torch.from_numpy(action_mask).bool()

        # Pad to unified dimensions
        unified_action_dim = ActionFeature.ALL_FEATURES[1]  # 192
        unified_state_dim = StateFeature.ALL_FEATURES[1]    # 212

        unified_state, unified_state_mask = pad_state_robot(
            state=norm_state_tensor,
            state_mask=state_mask_tensor,
            state_dim=state_dim,
            unified_state_dim=unified_state_dim,
        )

        _, unified_action_mask = pad_action(
            actions=None,
            action_mask=action_mask_tensor,
            action_dim=state_dim,
            unified_action_dim=unified_action_dim,
        )

        extra_data = {}
        return unified_state, unified_state_mask, unified_action_mask, extra_data

    def postprocess_action(
        self,
        norm_action: np.ndarray,
        extra_data: dict,
    ) -> np.ndarray:
        """
        Post-process normalized action from model.

        For 6D position format, simple transfer and denormalization.
        """
        norm_action_tensor = torch.from_numpy(norm_action).float()
        norm_robot_action = transfer_human_to_xhand(norm_action_tensor, use_end_pose=True)
        unnorm_action = self.normalizer.unnormalize_action(norm_robot_action.numpy())

        return unnorm_action
