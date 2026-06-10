"""
MuseVLA Inference with 7D Joint Angle Output

Output format: 19D per hand (7D arm joints + 12D hand joints)
- Input state: 19D (right hand only)
- Output action: (chunk_size, 19) right hand actions

Note: The 7D format includes an extra 1-DoF compared to 6D (6D EEF + 1D extra).
This extra dimension requires special handling during state/action processing.
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


class MuseVLAInference7DJointAngle(MuseVLAInferenceBase):
    """
    MuseVLA inference with 7D joint angle output.

    State/Action format per hand: [7D arm joints, 12D hand joints] = 19D
    Total (both hands): 38D

    The extra 1-DoF (7th joint) requires special handling:
    - It's extracted before padding to unified dimensions
    - It's stored in ROBOT_LEFT_EXTRA_1D and ROBOT_RIGHT_EXTRA_1D slots
    - It's restored after transfer from human format
    """

    STATE_DIM = 19  # Per hand
    ACTION_DIM = 19  # Per hand
    FULL_STATE_DIM = 38  # Both hands
    FULL_ACTION_DIM = 38  # Both hands

    def convert_input_state(self, state: np.ndarray) -> np.ndarray:
        """
        Convert client's 19D right-hand state to 38D full state.

        Client sends right hand 19D: [7D arm joints, 12D hand joints]
        Convert to 38D: [19D left (zeros), 19D right]
        """
        state_38d = np.zeros(38, dtype=np.float32)
        state_38d[19:38] = state
        return state_38d

    def convert_output_action(self, action: np.ndarray) -> np.ndarray:
        """
        Convert model's 38D action output to 19D right-hand action.

        Model outputs 38D: [19D left, 19D right]
        Extract right hand 19D: [7D arm joints, 12D hand joints]
        """
        return action[:, 19:38]

    def prepare_state_and_masks(
        self,
        state_full: np.ndarray,
        state_mask: Optional[np.ndarray],
        action_mask: Optional[np.ndarray],
    ) -> tuple:
        """
        Prepare state tensor and masks for VLA prediction.

        For 7D joint angle format, extract the extra 1-DoF dimensions
        and store them separately for later restoration.
        """
        state_dim = self.FULL_STATE_DIM  # 38

        if state_mask is None:
            state_mask = np.zeros(state_dim, dtype=bool)
            state_mask[19:38] = True  # Enable right hand only

        if action_mask is None:
            action_mask = np.zeros((self.chunk_size, state_dim), dtype=bool)
            action_mask[:, 19:38] = True  # Predict right hand only

        # Normalize state
        norm_state = self.normalizer.normalize_state(state_full)

        # Convert to tensors
        norm_state_tensor = torch.from_numpy(norm_state).float()
        state_mask_tensor = torch.from_numpy(state_mask).bool()
        action_mask_tensor = torch.from_numpy(action_mask).bool()

        # Extract extra 1-DoF dimensions from normalized state (for 7-DoF arms)
        left_data_dim = 19
        right_data_dim = 38

        # Extract left extra_1d (index 0)
        extra_1d_left_state = norm_state_tensor[0]
        extra_1d_left_state_mask = state_mask_tensor[0]
        extra_1d_left_mask = action_mask_tensor[:, 0]

        # Extract right extra_1d (index 19)
        extra_1d_right_state = norm_state_tensor[19]
        extra_1d_right_state_mask = state_mask_tensor[19]
        extra_1d_right_mask = action_mask_tensor[:, 19]

        # Remove extra_1d from tensors (convert 38D -> 36D)
        norm_state_tensor = torch.cat([
            norm_state_tensor[1:left_data_dim],
            norm_state_tensor[left_data_dim + 1:right_data_dim],
        ])
        state_mask_tensor = torch.cat([
            state_mask_tensor[1:left_data_dim],
            state_mask_tensor[left_data_dim + 1:right_data_dim],
        ])
        action_mask_tensor = torch.cat([
            action_mask_tensor[:, 1:left_data_dim],
            action_mask_tensor[:, left_data_dim + 1:right_data_dim],
        ], dim=1)

        # Pad to unified dimensions
        unified_action_dim = ActionFeature.ALL_FEATURES[1]  # 192
        unified_state_dim = StateFeature.ALL_FEATURES[1]    # 212
        clipped_state_dim = norm_state_tensor.shape[0]      # 36
        clipped_action_dim = action_mask_tensor.shape[1]    # 36

        unified_state, unified_state_mask = pad_state_robot(
            state=norm_state_tensor,
            state_mask=state_mask_tensor,
            state_dim=clipped_state_dim,
            unified_state_dim=unified_state_dim,
        )

        _, unified_action_mask = pad_action(
            actions=None,
            action_mask=action_mask_tensor,
            action_dim=clipped_action_dim,
            unified_action_dim=unified_action_dim,
        )

        # Add back the extracted 7-DoF extra dimensions to unified tensors
        unified_state[StateFeature.ROBOT_LEFT_EXTRA_1D[0]:StateFeature.ROBOT_LEFT_EXTRA_1D[1]] = extra_1d_left_state
        unified_state_mask[StateFeature.ROBOT_LEFT_EXTRA_1D[0]:StateFeature.ROBOT_LEFT_EXTRA_1D[1]] = extra_1d_left_state_mask
        unified_action_mask[:, ActionFeature.ROBOT_LEFT_EXTRA_1D[0]:ActionFeature.ROBOT_LEFT_EXTRA_1D[1]] = extra_1d_left_mask.unsqueeze(-1)

        unified_state[StateFeature.ROBOT_RIGHT_EXTRA_1D[0]:StateFeature.ROBOT_RIGHT_EXTRA_1D[1]] = extra_1d_right_state
        unified_state_mask[StateFeature.ROBOT_RIGHT_EXTRA_1D[0]:StateFeature.ROBOT_RIGHT_EXTRA_1D[1]] = extra_1d_right_state_mask
        unified_action_mask[:, ActionFeature.ROBOT_RIGHT_EXTRA_1D[0]:ActionFeature.ROBOT_RIGHT_EXTRA_1D[1]] = extra_1d_right_mask.unsqueeze(-1)

        extra_data = {}
        return unified_state, unified_state_mask, unified_action_mask, extra_data

    def postprocess_action(
        self,
        norm_action: np.ndarray,
        extra_data: dict,
    ) -> np.ndarray:
        """
        Post-process normalized action from model.

        For 7D joint angle format, transfer and denormalize.
        """
        norm_action_tensor = torch.from_numpy(norm_action).float()
        norm_robot_action = transfer_human_to_xhand(norm_action_tensor)
        unnorm_action = self.normalizer.unnormalize_action(norm_robot_action.numpy())

        return unnorm_action
