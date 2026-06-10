"""
robot_dataset.py

Dataset implementation for robot manipulation data (e.g., XHand teleoperation).
Handles loading, processing, and normalization of robot state and action trajectories.
"""

import json
import os
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R

from musevla.datasets.augment_utils import apply_color_augmentation
from musevla.datasets.dataset_utils import (
    ActionFeature,
    StateFeature,
    calculate_fov,
    compute_new_intrinsics_crop,
    compute_new_intrinsics_resize,
)
from musevla.datasets.human_dataset import pad_action
from musevla.utils.data_utils import GaussianNormalizer, read_dataset_statistics


# Mapping from XHand joint indices to human hand model indices
# Format: (xhand_idx, human_idx, sign_multiplier)
XHAND_HUMAN_MAPPING = [
    (10, 8, 1),
    (11, 14, 1),
    (12, 17, 1),
    (13, 23, 1),
    (16, 26, 1),
    (17, 32, 1),
    (14, 35, 1),
    (15, 41, 1),
    (7, 43, -1),
    (6, 44, 1),
    (8, 46, -1),
    (9, 7, 1),
]

AGIHAND_HUMAN_MAPPING = [
    (10, 35, 1),
    (11, 26, 1),
    (7, 46, -1),
    (6, 44, 1),
    (8, 8, 1),
    (9, 17, 1),
]

GRIPPER_HUMAN_MAPPING = [
    (6, 44, 1),
]


class RoboDatasetCore(object):
    """
    Core dataset class for robot manipulation data.
    
    Handles loading and processing of robot teleoperation trajectories,
    including state/action sequences, images, and camera parameters.
    """
    
    def __init__(
        self,
        root_dir: str,
        statistics_path: Optional[str] = None,
        action_past_window_size: int = 0,
        action_future_window_size: int = 16,
        image_past_window_size: int = 0,
        image_future_window_size: int = 0,
        load_images: bool = True
    ):
        """
        Initialize robot dataset core.
        
        Args:
            root_dir: Root directory containing robot data
            statistics_path: Path to normalization statistics JSON file
            action_past_window_size: Number of past action frames to include
            action_future_window_size: Number of future action frames to predict
            image_past_window_size: Number of past image frames to include
            image_future_window_size: Number of future image frames to include
            load_images: Whether to load image data
        """
        self.root = root_dir
        self.action_past_window_size = action_past_window_size
        self.action_future_window_size = action_future_window_size
        self.image_past_window_size = image_past_window_size
        self.image_future_window_size = image_future_window_size
        self.load_images = load_images
        
        # Load normalization statistics if provided
        if statistics_path is not None:
            self.data_statistics = read_dataset_statistics(statistics_path)
        else:
            self.data_statistics = None
        
        # TODO: Implement data loading
        # - Load instruction JSON
        # - Build sample index
        
        raise NotImplementedError("Data loading logic needs to be implemented")

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def set_global_data_statistics(self, global_data_statistics: dict):
        """
        Set global normalization statistics and initialize normalizer.
        
        Args:
            global_data_statistics: Dictionary containing mean/std for state/action
        """
        self.global_data_statistics = global_data_statistics
        if not hasattr(self, 'gaussian_normalizer'):
            self.gaussian_normalizer = GaussianNormalizer(self.global_data_statistics)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary containing:
                - image_list: List of RGB images
                - action_list: Action sequence array
                - current_state: Current robot state
                - action_mask: Mask indicating valid actions
                - current_state_mask: Mask indicating valid state dimensions
                - fov: Field of view (horizontal, vertical)
                - intrinsics: Camera intrinsic matrix
                - instruction: Text instruction
        """
        # TODO: Implement data loading for single sample
        # - Load image(s)
        # - Load state and action sequences
        # - Apply augmentation if needed
        # - Compute FOV from intrinsics
        raise NotImplementedError("Sample loading logic needs to be implemented")

    def transform_trajectory(
        self,
        sample_dict: dict = None,
        normalization: bool = True,
    ):
        """Pad action and state dimensions to match a unified size."""

        action_np = sample_dict["action_list"]
        state_np = sample_dict["current_state"]
       
        action_dim = action_np.shape[1]
        state_dim = state_np.shape[0]
        if normalization:
            # Normalize left and right hand actions and states separately
            action_np = self.gaussian_normalizer.normalize_action(action_np)
            state_np = self.gaussian_normalizer.normalize_state(state_np)

        # ===== Pad to unified dimensions =====
        unified_action_dim = ActionFeature.ALL_FEATURES[1]   # 192
        unified_state_dim = StateFeature.ALL_FEATURES[1]     # 212

        unified_action, unified_action_mask = pad_action(
            action_np,
            sample_dict["action_mask"],
            action_dim,
            unified_action_dim
        )

        unified_state, unified_state_mask = pad_state_robot(
            state_np,
            sample_dict["current_state_mask"],
            state_dim,
            unified_state_dim
        )

        human_state, human_state_mask, human_action, human_action_mask = transfer_xhand_to_human(
            unified_state, unified_state_mask,
            unified_action, unified_action_mask
        )

        sample_dict["action_list"] = human_action
        sample_dict["action_mask"] = human_action_mask
        sample_dict["current_state"] = human_state
        sample_dict["current_state_mask"] = human_state_mask

        return sample_dict

def pad_state_robot(
    state: torch.Tensor,
    state_mask: torch.Tensor,
    state_dim: int,
    unified_state_dim: int,
    has_left_arm: bool = True
):
    """
    Convert robot state to unified dual-arm representation and pad to standard size.
    
    This function:
    1. Ensures state_mask has same dimensions as state (per-dimension mask)
    2. Masks invalid state dimensions
    3. Converts single-arm (right-only) to dual-arm format by placing data in right half
    4. Pads to unified_state_dim with zeros

    Args:
        state (Tensor): original state tensor, shape [state_dim]
        state_mask (Tensor): state mask, shape [state_dim] (per-dimension)
        state_dim (int): original state dimension
        unified_state_dim (int): target padded state dimension (must be even)
        has_left_arm (bool): whether dataset has left arm data (False for single-arm datasets)

    Returns:
        Tuple[Tensor, Tensor]: 
            padded state [unified_state_dim],
            padded state_mask [unified_state_dim]
    """
    # Handle both numpy arrays and tensors
    if isinstance(state, torch.Tensor):
        current_state = state.float()
    else:
        current_state = torch.tensor(state, dtype=torch.float32)
    
    if isinstance(state_mask, torch.Tensor):
        current_state_mask = state_mask.bool()
    else:
        current_state_mask = torch.tensor(state_mask, dtype=torch.bool)
    
    # Ensure mask has same dimension as state
    if current_state_mask.shape[0] != state_dim:
        raise ValueError(f"state_mask shape {current_state_mask.shape} does not match state_dim {state_dim}")

    # Mask out invalid state dimensions
    current_state_masked = current_state * current_state_mask.to(current_state.dtype)

    # Initialize output tensors with zeros (for padding)
    padded_state = torch.zeros(unified_state_dim, dtype=current_state.dtype)
    padded_mask = torch.zeros(unified_state_dim, dtype=torch.bool)

    if has_left_arm:
        # Dual-arm format: data is [left_data, right_data]
        # Split at midpoint and place in respective halves
        left_data_size = state_dim // 2
        right_data_size = state_dim - left_data_size
        
        # Place left arm data in left half of unified dimension
        padded_state[:left_data_size] = current_state_masked[:left_data_size]
        padded_mask[:left_data_size] = current_state_mask[:left_data_size]

        # Place right arm data in right half of unified dimension (starting from midpoint)
        right_start = state_dim // 2
        padded_state[right_start:right_start+right_data_size] = current_state_masked[left_data_size:]
        padded_mask[right_start:right_start+right_data_size] = current_state_mask[left_data_size:]
    else:
        # Single-arm format: all data is right arm only
        # Leave space for left arm (zeros), place right arm at offset state_dim
        right_start = state_dim
        padded_state[right_start:right_start+state_dim] = current_state_masked
        padded_mask[right_start:right_start+state_dim] = current_state_mask

    return padded_state, padded_mask

def transfer_xhand_to_human(
    unified_state: torch.Tensor,
    unified_state_mask: torch.Tensor,
    unified_action: Optional[torch.Tensor],
    unified_action_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Transfer XHand robot format to human hand model format.
    
    Maps robot joint angles and end-effector poses to the human hand
    representation used by the model.
    
    Args:
        unified_state: Robot state tensor, shape (212,) - required
        unified_state_mask: State validity mask, shape (212,) - required
        unified_action: Robot action sequence, shape (T, 192) - can be None
        unified_action_mask: Action validity mask, shape (T, 192) - required
        
    Returns:
        Tuple of (human_state, human_state_mask, human_action, human_action_mask)
        human_action will be None if unified_action is None
    """
    
    # Initialize output tensors for state (always required)
    human_state = torch.zeros(unified_state.shape[0], dtype=torch.float32)
    human_state_mask = torch.zeros(unified_state_mask.shape[0], dtype=torch.bool)
    
    # Preserve ROBOT_*_EXTRA_1D values (indices 102-103) for 7-DoF arms
    human_state[102:104] = unified_state[102:104]
    human_state_mask[102:104] = unified_state_mask[102:104]
    
    # Transfer left hand end-effector 6-DoF pose (translation + rotation)
    human_state[0:6] = unified_state[0:6]
    human_state_mask[0:6] = unified_state_mask[0:6]

    # Transfer right hand end-effector 6-DoF pose
    human_state[51:57] = unified_state[18:24]
    human_state_mask[51:57] = unified_state_mask[18:24]

    for src, dst, sign in XHAND_HUMAN_MAPPING:
        human_state[dst] = sign * unified_state[src]
        human_state[dst+51] = sign * unified_state[src+18]

    # Set state mask strategy: if hand is active, set entire hand dimensions to active
    human_state_mask[0:51] = unified_state_mask[0]  # scalar bool for left hand
    human_state_mask[51:102] = unified_state_mask[18]  # scalar bool for right hand

    # Initialize action mask (always required, even if action is None)
    human_action_mask = torch.zeros((unified_action_mask.shape[0], 192), dtype=torch.bool)
    
    # Preserve ROBOT_*_EXTRA_1D mask values (indices 102-103) for 7-DoF arms
    human_action_mask[:, 102:104] = unified_action_mask[:, 102:104]

    # Set action mask ONLY at indices where data is actually mapped (6D EEF + 12 mapped joints per hand)
    # NOT the entire 51-dim hand block — unmapped indices have zero data and should not contribute to loss
    left_hand_active = unified_action_mask[:, 0]  # (T,) bool array
    right_hand_active = unified_action_mask[:, 18]  # (T,) bool array

    # Left hand 6D EEF pose (indices 0:6)
    human_action_mask[:, 0:6] = left_hand_active.unsqueeze(1).expand(-1, 6)
    # Right hand 6D EEF pose (indices 51:57)
    human_action_mask[:, 51:57] = right_hand_active.unsqueeze(1).expand(-1, 6)

    # Mapped joint indices only (12 per hand from XHAND_HUMAN_MAPPING)
    for src, dst, sign in XHAND_HUMAN_MAPPING:
        human_action_mask[:, dst] = left_hand_active        # left hand joint
        human_action_mask[:, dst + 51] = right_hand_active  # right hand joint

    # Handle action data (can be None)
    if unified_action is None:
        return human_state, human_state_mask, None, human_action_mask
    
    # Initialize and process action data
    human_action = torch.zeros((unified_action.shape[0], 192), dtype=torch.float32)
    
    # Preserve ROBOT_*_EXTRA_1D values (indices 102-103) for 7-DoF arms
    human_action[:, 102:104] = unified_action[:, 102:104]
    
    human_action[:, 0:6] = unified_action[:, 0:6]
    human_action[:, 51:57] = unified_action[:, 18:24]

    for src, dst, sign in XHAND_HUMAN_MAPPING:
        human_action[:, dst] = sign * unified_action[:, src]
        human_action[:, dst+51] = sign * unified_action[:, src+18]

    return human_state, human_state_mask, human_action, human_action_mask


def transfer_gripper_to_human(
    unified_state: torch.Tensor,
    unified_state_mask: torch.Tensor,
    unified_action: Optional[torch.Tensor],
    unified_action_mask: torch.Tensor,
    use_ee_action: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Transfer gripper robot format to human hand model format.
    
    For simple gripper robots (e.g., MolmoAct, InternA1), maps end-effector poses
    and gripper state to the human hand representation.
    Gripper format: 7 dims per hand (6 EEF + 1 gripper), left at 0-6, right at 7-13
    
    Args:
        unified_state: Robot state tensor, shape (212,) - required
        unified_state_mask: State validity mask, shape (212,) - required
        unified_action: Robot action sequence, shape (T, 192) - can be None
        unified_action_mask: Action validity mask, shape (T, 192) - required
        
    Returns:
        Tuple of (human_state, human_state_mask, human_action, human_action_mask)
        human_action will be None if unified_action is None
    """
    # Initialize output tensors for state (always required)
    human_state = torch.zeros(unified_state.shape[0], dtype=torch.float32)
    human_state_mask = torch.zeros(unified_state_mask.shape[0], dtype=torch.bool)
    
    # Preserve ROBOT_*_EXTRA_1D values (indices 102-103) for 7-DoF arms
    human_state[102:104] = unified_state[102:104]
    human_state_mask[102:104] = unified_state_mask[102:104]
    
    # Transfer left hand end-effector 6-DoF pose (translation + rotation)
    # Gripper: left hand at indices 0-6 (6 EEF + 1 gripper)
    human_state[0:6] = unified_state[0:6]
    human_state_mask[0:6] = unified_state_mask[0:6]

    # Transfer right hand end-effector 6-DoF pose
    # Gripper: right hand at indices 7-13 (6 EEF + 1 gripper)
    human_state[51:57] = unified_state[7:13]
    human_state_mask[51:57] = unified_state_mask[7:13]

    # Map gripper state using GRIPPER_HUMAN_MAPPING
    # src index 6 = left gripper, src index 13 = right gripper (7+6)
    for src, dst, sign in GRIPPER_HUMAN_MAPPING:
        human_state[dst] = sign * unified_state[src]
        human_state[dst+51] = sign * unified_state[src+7]  # right hand offset is 7 for gripper

    # Set state mask strategy: if hand is active, set entire hand dimensions to active
    human_state_mask[0:51] = unified_state_mask[0]  # scalar bool for left hand
    human_state_mask[51:102] = unified_state_mask[7]  # scalar bool for right hand (offset 7)

    # Initialize action mask (always required, even if action is None)
    human_action_mask = torch.zeros((unified_action_mask.shape[0], 192), dtype=torch.bool)
    
    # Preserve ROBOT_*_EXTRA_1D mask values (indices 102-103) for 7-DoF arms
    human_action_mask[:, 102:104] = unified_action_mask[:, 102:104]

    # Set action mask ONLY at indices where data is actually mapped (6D EEF + 1 mapped joint per hand)
    # NOT the entire 51-dim hand block — unmapped indices have zero data and should not contribute to loss
    left_hand_active = unified_action_mask[:, 0]  # (T,) bool array
    right_hand_active = unified_action_mask[:, 7]  # (T,) bool array, offset 7 for gripper

    # Left hand 6D EEF pose (indices 0:6)
    human_action_mask[:, 0:6] = left_hand_active.unsqueeze(1).expand(-1, 6)
    # Right hand 6D EEF pose (indices 51:57)
    human_action_mask[:, 51:57] = right_hand_active.unsqueeze(1).expand(-1, 6)

    # Mapped joint indices only (1 per hand from GRIPPER_HUMAN_MAPPING)
    if use_ee_action:
        for src, dst, sign in GRIPPER_HUMAN_MAPPING:
            human_action_mask[:, dst] = left_hand_active        # left hand joint
            human_action_mask[:, dst + 51] = right_hand_active  # right hand joint

    # Handle action data (can be None)
    if unified_action is None:
        return human_state, human_state_mask, None, human_action_mask
    
    # Initialize and process action data
    human_action = torch.zeros((unified_action.shape[0], 192), dtype=torch.float32)
    
    # Preserve ROBOT_*_EXTRA_1D values (indices 102-103) for 7-DoF arms
    human_action[:, 102:104] = unified_action[:, 102:104]
    
    human_action[:, 0:6] = unified_action[:, 0:6]
    human_action[:, 51:57] = unified_action[:, 7:13]  # right hand offset is 7

    for src, dst, sign in GRIPPER_HUMAN_MAPPING:
        human_action[:, dst] = sign * unified_action[:, src]
        human_action[:, dst+51] = sign * unified_action[:, src+7]  # right hand offset is 7

    return human_state, human_state_mask, human_action, human_action_mask


def transfer_agihand_to_human(
    unified_state: torch.Tensor,
    unified_state_mask: torch.Tensor,
    unified_action: Optional[torch.Tensor],
    unified_action_mask: torch.Tensor,
    use_ee_action: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Transfer AgiHand robot format to human hand model format.
    
    For AgiHand robots (subset of Agibot dataset), maps joint angles and
    end-effector poses to the human hand representation.
    AgiHand format: 12 dims per hand (6 EEF + 6 joints), left at 0-11, right at 12-23
    
    Args:
        unified_state: Robot state tensor, shape (212,) - required
        unified_state_mask: State validity mask, shape (212,) - required
        unified_action: Robot action sequence, shape (T, 192) - can be None
        unified_action_mask: Action validity mask, shape (T, 192) - required
        
    Returns:
        Tuple of (human_state, human_state_mask, human_action, human_action_mask)
        human_action will be None if unified_action is None
    """
    # Initialize output tensors for state (always required)
    human_state = torch.zeros(unified_state.shape[0], dtype=torch.float32)
    human_state_mask = torch.zeros(unified_state_mask.shape[0], dtype=torch.bool)
    
    # Preserve ROBOT_*_EXTRA_1D values (indices 102-103) for 7-DoF arms
    human_state[102:104] = unified_state[102:104]
    human_state_mask[102:104] = unified_state_mask[102:104]
    
    # Transfer left hand end-effector 6-DoF pose (translation + rotation)
    # AgiHand: left hand at indices 0-11 (6 EEF + 6 joints)
    human_state[0:6] = unified_state[0:6]
    human_state_mask[0:6] = unified_state_mask[0:6]

    # Transfer right hand end-effector 6-DoF pose
    # AgiHand: right hand at indices 12-23 (6 EEF + 6 joints)
    human_state[51:57] = unified_state[12:18]
    human_state_mask[51:57] = unified_state_mask[12:18]

    # Map AgiHand joints using AGIHAND_HUMAN_MAPPING
    # Left hand joints at 6-11, right hand joints at 18-23 (12+6)
    for src, dst, sign in AGIHAND_HUMAN_MAPPING:
        human_state[dst] = sign * unified_state[src]
        human_state[dst+51] = sign * unified_state[src+12]  # right hand offset is 12 for agihand

    # Set state mask strategy: if hand is active, set entire hand dimensions to active
    human_state_mask[0:51] = unified_state_mask[0]  # scalar bool for left hand
    human_state_mask[51:102] = unified_state_mask[12]  # scalar bool for right hand (offset 12)

    # Initialize action mask (always required, even if action is None)
    human_action_mask = torch.zeros((unified_action_mask.shape[0], 192), dtype=torch.bool)
    
    # Preserve ROBOT_*_EXTRA_1D mask values (indices 102-103) for 7-DoF arms
    human_action_mask[:, 102:104] = unified_action_mask[:, 102:104]

    # Set action mask ONLY at indices where data is actually mapped (6D EEF + 6 mapped joints per hand)
    # NOT the entire 51-dim hand block — unmapped indices have zero data and should not contribute to loss
    left_hand_active = unified_action_mask[:, 0]  # (T,) bool array
    right_hand_active = unified_action_mask[:, 12]  # (T,) bool array, offset 12 for agihand

    # Left hand 6D EEF pose (indices 0:6)
    human_action_mask[:, 0:6] = left_hand_active.unsqueeze(1).expand(-1, 6)
    # Right hand 6D EEF pose (indices 51:57)
    human_action_mask[:, 51:57] = right_hand_active.unsqueeze(1).expand(-1, 6)

    # Mapped joint indices only (6 per hand from AGIHAND_HUMAN_MAPPING)
    if use_ee_action:
        for src, dst, sign in AGIHAND_HUMAN_MAPPING:
            human_action_mask[:, dst] = left_hand_active        # left hand joint
            human_action_mask[:, dst + 51] = right_hand_active  # right hand joint

    # Handle action data (can be None)
    if unified_action is None:
        return human_state, human_state_mask, None, human_action_mask
    
    # Initialize and process action data
    human_action = torch.zeros((unified_action.shape[0], 192), dtype=torch.float32)
    
    # Preserve ROBOT_*_EXTRA_1D values (indices 102-103) for 7-DoF arms
    human_action[:, 102:104] = unified_action[:, 102:104]
    
    human_action[:, 0:6] = unified_action[:, 0:6]
    human_action[:, 51:57] = unified_action[:, 12:18]  # right hand offset is 12

    for src, dst, sign in AGIHAND_HUMAN_MAPPING:
        human_action[:, dst] = sign * unified_action[:, src]
        human_action[:, dst+51] = sign * unified_action[:, src+12]  # right hand offset is 12

    return human_state, human_state_mask, human_action, human_action_mask


def transfer_human_to_xhand(
    human_action: torch.Tensor,
    use_end_pose: bool = False,
    arm_ik_solver: Optional[object] = None,
    current_joint_angles: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Transfer human hand model format back to XHand robot format.
    Maps human hand representation to robot joint angles and end-effector poses.
    This is the inverse operation of transfer_xhand_to_human for actions only.
    
    Args:
        human_action: Human action sequence, shape (T, 192)
        use_end_pose: If True, convert 6D end pose (indices 0-5, 51-56) to 7D joint angles using IK
                      If False, use joint angles directly from indices 102-103
        arm_ik_solver: IK solver instance (required if use_end_pose=True)
        current_joint_angles: Current 7-DoF joint angles for IK initialization, shape (7,) (optional)
    
    Returns:
        xhand_action: Robot action sequence, shape (T, 38) - 19 dims per hand (1 extra DoF + 6 EEF + 12 joints)
                      OR shape (T, 36) - 18 dims per hand (6 EEF + 12 joints) if use_end_pose=True and no IK solver provided
    """
    T = human_action.shape[0]
    
    if use_end_pose:
        # Position-based mode: use 6D pose directly, may need IK for joint angles
        if arm_ik_solver is not None:
            # Use IK to convert 6D pose to 7D joint angles
            # Initialize output tensor: 19 dims per hand (1 extra + 6 EEF + 12 joints)
            xhand_action = torch.zeros((T, 38), dtype=torch.float32)
            
            # Process right hand (currently only right hand is supported for IK)
            # Extract 6D pose from human_action [51:57] -> [x, y, z, rot_x, rot_y, rot_z]
            for t in range(T):
                # Right hand 6D pose
                pose_6d = human_action[t, 51:57].numpy()  # [x, y, z, rot_x, rot_y, rot_z]
                
                # TODO: Convert 6D pose to 4x4 transformation matrix for IK
                # This requires camera-to-arm coordinate transformation
                # For now, we'll use the pose directly and assume it's in the correct frame
                
                # Construct 4x4 transformation matrix
                from scipy.spatial.transform import Rotation as R
                position = pose_6d[:3]  # x, y, z
                rotation_vec = pose_6d[3:]  # rot_x, rot_y, rot_z
                
                # Convert rotation vector to rotation matrix
                rotation = R.from_rotvec(rotation_vec)
                rotation_matrix = rotation.as_matrix()
                
                # Build 4x4 transformation matrix
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = rotation_matrix
                transform_matrix[:3, 3] = position
                
                # Use IK to compute joint angles
                init_angles = current_joint_angles if t == 0 and current_joint_angles is not None else None
                joint_angles, solved, _ = arm_ik_solver.ik(transform_matrix, init_angles=init_angles)
                
                if solved:
                    # Store joint angles for next iteration
                    current_joint_angles = joint_angles
                    # Set all 7D joint angles for right arm (indices 19-26)
                    xhand_action[t, 19:26] = torch.tensor(joint_angles, dtype=torch.float32)
                else:
                    # IK failed, use zeros or previous solution
                    if t > 0:
                        xhand_action[t, 19:26] = xhand_action[t-1, 19:26]
                    # Log warning
                    import logging
                    logging.warning(f"IK failed at timestep {t}, using fallback")
            
            # Transfer hand joint angles using reverse mapping (right hand only)
            # Hand joints start at index 26 (after 7 arm joints at 19-25)
            for src, dst, sign in XHAND_HUMAN_MAPPING:
                # Right hand: human -> xhand (with +26 offset: 19 for right hand start + 7 for arm joints)
                xhand_action[:, src+20] = sign * human_action[:, dst+51]
            
            # Left hand - use joint angles from human_action (indices 102-103 for extra DoF)
            # Note: For left hand in IK mode, we still use the original joint representation
            xhand_action[:, 0:1] = human_action[:, 102:103]  # Extra DoF
            xhand_action[:, 1:7] = human_action[:, 0:6]  # 6D pose (or first 6 joints if available)
            for src, dst, sign in XHAND_HUMAN_MAPPING:
                xhand_action[:, src+1] = sign * human_action[:, dst]
                
        else:
            # No IK solver: output 18 dims per hand (6 EEF + 12 joints, no extra DoF)
            xhand_action = torch.zeros((T, 36), dtype=torch.float32)

            # Transfer left hand end-effector 6-DoF pose
            xhand_action[:, 0:6] = human_action[:, 0:6]
            # Transfer right hand end-effector 6-DoF pose
            xhand_action[:, 18:24] = human_action[:, 51:57]

            # Transfer joint angles using reverse mapping
            # In 18D format: joints are at indices 6-17 (left) and 24-35 (right)
            # XHAND_HUMAN_MAPPING src indices (6-17) map directly without offset for left hand
            # For right hand: offset is +18 (not +24, since no extra_1d)
            for src, dst, sign in XHAND_HUMAN_MAPPING:
                # Left hand: human -> xhand (src is already 6-17 range)
                xhand_action[:, src] = sign * human_action[:, dst]
                # Right hand: human -> xhand (with +18 offset for right hand start)
                xhand_action[:, src + 18] = sign * human_action[:, dst + 51]
    else:
        # Joint-based mode: use original implementation with 7D joint angles
        # Initialize output tensor: 19 dims per hand (1 extra + 6 EEF + 12 joints)
        xhand_action = torch.zeros((T, 38), dtype=torch.float32)
        # Transfer left hand extra 1-DoF (arm joint) - FIRST
        xhand_action[:, 0:1] = human_action[:, 102:103]
        # Transfer left hand end-effector 6-DoF pose (translation + rotation)
        xhand_action[:, 1:7] = human_action[:, 0:6]
        # Transfer right hand extra 1-DoF (arm joint) - FIRST
        xhand_action[:, 19:20] = human_action[:, 103:104]
        # Transfer right hand end-effector 6-DoF pose
        xhand_action[:, 20:26] = human_action[:, 51:57]
        
        # Transfer joint angles using reverse mapping
        for src, dst, sign in XHAND_HUMAN_MAPPING:
            # Left hand: human -> xhand (with +1 offset for extra DoF at beginning)
            xhand_action[:, src+1] = sign * human_action[:, dst]
            # Right hand: human -> xhand (with +20 offset: 19 for right hand + 1 for extra DoF at beginning)
            xhand_action[:, src+20] = sign * human_action[:, dst+51]
    
    return xhand_action