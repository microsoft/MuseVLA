
import numpy as np
import json
import sys
from tqdm import tqdm
import argparse
import os
from pathlib import Path
import cv2
import h5py
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.spatial.transform import Rotation as R  # For proper rotation delta calculation

import pyarrow.parquet as pq


def is_video_valid(video_path: str, timeout=30) -> bool:
    """
    Validate a video by fully decoding it with ffmpeg.
    Any stderr output means the video is invalid.
    """
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-i", video_path,
        "-f", "null",
        "-"
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True
        )
        return result.stderr.strip() == ""
    except Exception:
        return False


def prefilter_videos(video_paths):
    """
    Validate a list of video paths sequentially.
    
    Args:
        video_paths: List of video file paths to validate
    
    Returns:
        Tuple of (valid_paths, invalid_paths)
    """
    valid = []
    invalid = []

    for path in tqdm(video_paths, desc="Validating videos"):
        if is_video_valid(path):
            valid.append(path)
        else:
            invalid.append(path)

    print(f"Valid videos   : {len(valid)}")
    print(f"Invalid videos : {len(invalid)}")

    return valid, invalid


def filter_and_cache_valid_videos(dataset_name, root_dir):
    """
    Filter valid videos and cache the invalid list to JSON.
    This should be run ONCE to create an invalid_videos.json cache file.
    
    Args:
        dataset_name: Name of the dataset
        root_dir: Root directory containing the dataset
    """
    root = Path(root_dir)
    
    print(f"\n{'='*60}")
    print(f"Filtering valid videos for dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Collect all video paths based on dataset type
    all_video_paths = []
    
    if dataset_name == 'vitra':
        metadata_file = root / "all_metadata_dist.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
            for item in metadata_list:
                video_path = root / item['relative_path'] / "video.mp4"
                if video_path.exists():
                    all_video_paths.append(str(video_path))
    
    elif 'molmoact' in dataset_name:
        split = dataset_name.split('_')[1]
        videos_dir = root / "train" / "videos"
        if videos_dir.exists():
            for chunk_dir in videos_dir.iterdir():
                if chunk_dir.is_dir():
                    rgb_dir = chunk_dir / "observations.rgb"
                    if rgb_dir.exists():
                        for video_file in rgb_dir.glob("*.mp4"):
                            all_video_paths.append(str(video_file))
    
    elif dataset_name == 'agibot':
        metadata_file = root / "task_info" / "extra_sensor.jsonl"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    task_id = item.get('task_id', '')
                    episode_id = item.get('episode_id', '')
                    if task_id and episode_id:
                        video_path = root / "observations_extract" / task_id / episode_id / "videos" / "head_color_h264.mp4"
                        if video_path.exists():
                            all_video_paths.append(str(video_path))
    
    print(f"Found {len(all_video_paths)} video files to validate")
    
    # Validate videos sequentially
    valid_video_paths, invalid_video_paths = prefilter_videos(all_video_paths)
    
    # Convert invalid videos to relative paths and save to JSON
    invalid_videos_relative = [str(Path(p).relative_to(root)) for p in invalid_video_paths]
    invalid_videos_set = set(invalid_videos_relative)
    
    cache_file = root / f"{dataset_name}_invalid_videos.json"
    with open(cache_file, 'w') as f:
        json.dump(sorted(list(invalid_videos_set)), f, indent=2)
    
    print(f"Invalid videos list saved to: {cache_file}")
    print(f"Total invalid videos: {len(invalid_videos_set)}")
    
    return invalid_videos_set


def compute_frame_counts(dataset_name, root_dir):
    """
    Compute and cache frame counts for all videos in a dataset.
    This should be run ONCE to create a JSON cache file.
    
    Args:
        dataset_name: Name of the dataset
        root_dir: Root directory containing the dataset
    """
    root = Path(root_dir)
    frame_counts = {}
    
    print(f"\n{'='*60}")
    print(f"Computing frame counts for dataset: {dataset_name}")
    print(f"{'='*60}")
    
    if dataset_name == 'vitra':
        frame_counts = compute_vitra_frame_counts(root)
    elif 'molmoact' in dataset_name:
        frame_counts = compute_molmoact_frame_counts(root, dataset_name)
    elif dataset_name == 'agibot':
        frame_counts = compute_agibot_frame_counts(root)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Save frame counts to JSON
    cache_file = root / f"{dataset_name}_frame_counts.json"
    with open(cache_file, 'w') as f:
        json.dump(frame_counts, f, indent=2)
    
    print(f"Frame counts saved to: {cache_file}")
    print(f"Total videos: {len(frame_counts)}")
    
    return frame_counts


def compute_vitra_frame_counts(root):
    """Compute frame counts for VITRA dataset."""
    metadata_file = root / "all_metadata_dist.json"
    if not metadata_file.exists():
        print(f"Warning: Metadata file not found: {metadata_file}")
        return {}
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
    
    frame_counts = {}
    for item in tqdm(metadata_list, desc="Counting VITRA frames"):
        video_path = root / item['relative_path'] / "video.mp4"
        if not video_path.exists():
            continue
        
        # Use relative path as key
        key = str(Path(item['relative_path']) / "video.mp4")
        
        cap = cv2.VideoCapture(str(video_path))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        frame_counts[key] = num_frames
    
    return frame_counts


def compute_molmoact_frame_counts(root, dataset_name):
    """Compute frame counts for MolmoAct dataset."""
    split = dataset_name.split('_')[1]  # household or tabletop
    frame_counts = {}
    
    for split_dir in ['train']:
        videos_dir = root / split_dir / "videos"
        if not videos_dir.exists():
            continue
        
        for chunk_dir in tqdm(sorted(videos_dir.iterdir()), desc=f"Counting MolmoAct {split} frames"):
            if not chunk_dir.is_dir():
                continue
            
            rgb_dir = chunk_dir / "observations.rgb"
            if not rgb_dir.exists():
                continue
            
            for video_file in rgb_dir.glob("*.mp4"):
                # Use relative path as key
                key = str(video_file.relative_to(root))
                
                cap = cv2.VideoCapture(str(video_file))
                num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                frame_counts[key] = num_frames
    
    return frame_counts


def compute_agibot_frame_counts(root):
    """Compute frame counts for Agibot dataset."""
    metadata_file = root / "task_info" / "extra_sensor.jsonl"
    if not metadata_file.exists():
        print(f"Warning: Metadata file not found: {metadata_file}")
        return {}
    
    frame_counts = {}
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Counting Agibot frames"):
            item = json.loads(line)
            
            task_id = item.get('task_id', '')
            episode_id = item.get('episode_id', '')
            
            if not task_id or not episode_id:
                continue
            
            video_path = root / "observations_extract" / task_id / episode_id / "videos" / "head_color_h264.mp4"
            if not video_path.exists():
                continue
            
            # Use relative path as key
            key = str(video_path.relative_to(root))
            
            cap = cv2.VideoCapture(str(video_path))
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            frame_counts[key] = num_frames
    
    return frame_counts


def compute_statistics_robot(dataset_name, root_dir, action_type='angle', use_end_pose=False, use_delta_pose=False, relative_to_t0=False, jump_frame=4, adaptive_jump_frame=2, threshold=0.087, **kwargs):
    """
    Compute mean and standard deviation of actions and states for robot datasets.
    
    Args:
        dataset_name: Name of the dataset (vitra, molmoact_household, molmoact_tabletop, agibot)
        root_dir: Root directory containing the dataset (also used as save folder)
        action_type: Type of action representation ('angle' or 'keypoints')
        use_end_pose: Whether to use 6D end pose (True) or 7D joint angles (False) for arm actions (VITRA only)
        use_delta_pose: Whether to use delta pose (pose_t+1 - pose_t) instead of absolute pose when use_end_pose=True
        relative_to_t0: Whether to compute actions relative to t0 frame (True) or frame-to-frame (False)
        jump_frame: Temporal downsampling stride (default: 4)
        adaptive_jump_frame: Smaller stride when hand command changes significantly (default: 2)
        threshold: Threshold for detecting significant hand command changes (default: 0.087)
    
    Returns:
        dict: A dictionary containing the mean and standard deviation
    """
    root = Path(root_dir)
    save_folder = root_dir  # Use root_dir as save folder
    
    if dataset_name == 'vitra' or dataset_name == 'real_robot':
        # Real robot uses same structure as VITRA
        action_data, state_data = load_vitra_data(root, use_end_pose=use_end_pose, use_delta_pose=use_delta_pose, relative_to_t0=relative_to_t0, jump_frame=jump_frame, adaptive_jump_frame=adaptive_jump_frame, threshold=threshold)
    elif dataset_name == 'real_robot_merged':
        extra_roots = kwargs.get('extra_roots', [])
        action_data, state_data = load_vitra_data(root, use_end_pose=use_end_pose, use_delta_pose=use_delta_pose, relative_to_t0=relative_to_t0, jump_frame=jump_frame, adaptive_jump_frame=adaptive_jump_frame, threshold=threshold, extra_roots=extra_roots)
    elif 'molmoact' in dataset_name:
        action_data, state_data = load_molmoact_data(root, dataset_name, use_end_pose=use_end_pose, use_delta_pose=use_delta_pose, relative_to_t0=relative_to_t0, jump_frame=jump_frame, adaptive_jump_frame=adaptive_jump_frame, threshold=threshold)
    elif dataset_name == 'agibot':
        # Agibot has two types: gripper and agihand
        action_data_gripper, state_data_gripper, action_data_agihand, state_data_agihand = load_agibot_data(root, use_end_pose=use_end_pose, use_delta_pose=use_delta_pose, relative_to_t0=relative_to_t0, jump_frame=jump_frame, adaptive_jump_frame=adaptive_jump_frame, threshold=threshold)

        # Build mode suffix matching vitra/real_robot convention
        agibot_suffix = ''
        if use_end_pose:
            if use_delta_pose:
                agibot_suffix += '_6dpose_delta'
            else:
                agibot_suffix += '_6dpose'
        if not relative_to_t0:
            agibot_suffix += '_NRt0'

        # Compute statistics for gripper version
        if len(action_data_gripper) > 0:
            stats_gripper = compute_stats_from_data(action_data_gripper, state_data_gripper, dataset_name + '_gripper' + agibot_suffix, action_type, save_folder)
            print(f"Gripper statistics: {len(action_data_gripper)} action samples, {len(state_data_gripper)} state samples")

        # Compute statistics for agihand version
        if len(action_data_agihand) > 0:
            stats_agihand = compute_stats_from_data(action_data_agihand, state_data_agihand, dataset_name + '_agihand' + agibot_suffix, action_type, save_folder)
            print(f"AgiHand statistics: {len(action_data_agihand)} action samples, {len(state_data_agihand)} state samples")

        return
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Add suffix for 6D pose mode and relative_to_t0 mode
    stats_dataset_name = dataset_name
    if (dataset_name in ('vitra', 'real_robot', 'real_robot_merged') or 'molmoact' in dataset_name) and use_end_pose:
        if use_delta_pose:
            stats_dataset_name = dataset_name + '_6dpose_delta'
        else:
            stats_dataset_name = dataset_name + '_6dpose'
    
    # Add '_NRt0' suffix if relative_to_t0 is False
    if (dataset_name in ('vitra', 'real_robot', 'real_robot_merged') or 'molmoact' in dataset_name) and not relative_to_t0:
        stats_dataset_name = stats_dataset_name + '_NRt0'
    
    compute_stats_from_data(action_data, state_data, stats_dataset_name, action_type, save_folder)


def compute_stats_from_data(action_data, state_data, dataset_name, action_type, save_folder):
    """Compute and save statistics from collected action and state data."""
    # Convert lists to numpy arrays
    actions = np.array(action_data, dtype=np.float32)  # [N, action_dim]
    states = np.array(state_data, dtype=np.float32)    # [M, state_dim]
    
    action_dim = actions.shape[1]
    state_dim = states.shape[1]
    
    # Determine if dataset has left arm based on dataset name
    # VITRA, MolmoAct, and real_robot are right-arm only
    # Agibot have dual arms
    base_dataset_name = dataset_name.replace('_gripper', '').replace('_agihand', '').replace('_6dpose_delta', '').replace('_6dpose', '').replace('_NRt0', '')
    has_left_arm = base_dataset_name == 'agibot'
    
    if has_left_arm:
        # Dual arm: split in half
        half_action_dim = action_dim // 2
        half_state_dim = state_dim // 2
        
        action_left = actions[:, :half_action_dim]
        action_right = actions[:, half_action_dim:]
        state_left = states[:, :half_state_dim]
        state_right = states[:, half_state_dim:]
    else:
        # Single arm (right only): use full dimensions for right, zeros for left
        action_left = np.zeros((actions.shape[0], action_dim), dtype=np.float32)
        action_right = actions
        state_left = np.zeros((states.shape[0], state_dim), dtype=np.float32)
        state_right = states
    
    # Compute statistics
    action_left_mean = np.mean(action_left, axis=0)
    action_left_std = np.std(action_left, axis=0)
    state_left_mean = np.mean(state_left, axis=0)
    state_left_std = np.std(state_left, axis=0)
    action_right_mean = np.mean(action_right, axis=0)
    action_right_std = np.std(action_right, axis=0)
    state_right_mean = np.mean(state_right, axis=0)
    state_right_std = np.std(state_right, axis=0)
    
    my_dict = {
        'dataset_name': f"{dataset_name}_{action_type}_statistics.json",
        'state_left_mean': state_left_mean.tolist(),
        'state_left_std': state_left_std.tolist(),
        'action_left_mean': action_left_mean.tolist(),
        'action_left_std': action_left_std.tolist(),
        'state_right_mean': state_right_mean.tolist(),
        'state_right_std': state_right_std.tolist(),
        'action_right_mean': action_right_mean.tolist(),
        'action_right_std': action_right_std.tolist(),
        'num_traj': len(states)
    }
    
    # Determine statistics path for this dataset
    base_dataset_name = dataset_name.replace('_gripper', '').replace('_agihand', '').replace('_6dpose_delta', '').replace('_6dpose', '').replace('_NRt0', '')
    if base_dataset_name in ("vitra", "real_robot", "real_robot_merged") or "molmoact" in base_dataset_name:
        stats_dir = Path(save_folder)
        # Use statistics_6dpose_delta.json for delta pose, statistics_6dpose.json for absolute 6D pose, otherwise statistics.json
        # Add '_NRt0' suffix if present in dataset_name
        if dataset_name.endswith("_6dpose_delta_NRt0"):
            suffix = "_6dpose_delta_NRt0"
        elif dataset_name.endswith("_6dpose_delta"):
            suffix = "_6dpose_delta"
        elif dataset_name.endswith("_6dpose_NRt0"):
            suffix = "_6dpose_NRt0"
        elif dataset_name.endswith("_6dpose"):
            suffix = "_6dpose"
        elif dataset_name.endswith("_NRt0"):
            suffix = "_NRt0"
        else:
            suffix = ""

        if base_dataset_name == "real_robot_merged":
            stats_file = stats_dir / f"statistics_merged{suffix}.json"
        else:
            stats_file = stats_dir / f"statistics{suffix}.json"
    elif base_dataset_name == "agibot":
        stats_dir = Path(save_folder)
        if '_gripper' in dataset_name:
            effector = '_gripper'
        elif '_agihand' in dataset_name:
            effector = '_agihand'
        else:
            effector = ''
        if dataset_name.endswith('_6dpose_delta_NRt0'):
            stats_file = stats_dir / f"statistics{effector}_6dpose_delta_NRt0.json"
        elif dataset_name.endswith('_6dpose_delta'):
            stats_file = stats_dir / f"statistics{effector}_6dpose_delta.json"
        elif dataset_name.endswith('_6dpose_NRt0'):
            stats_file = stats_dir / f"statistics{effector}_6dpose_NRt0.json"
        elif dataset_name.endswith('_6dpose'):
            stats_file = stats_dir / f"statistics{effector}_6dpose.json"
        elif dataset_name.endswith('_NRt0'):
            stats_file = stats_dir / f"statistics{effector}_NRt0.json"
        else:
            stats_file = stats_dir / f"statistics{effector}.json"
    else:
        stats_dir = Path(save_folder)
        stats_file = stats_dir / "statistics.json"  # Fallback
    
    os.makedirs(stats_dir, exist_ok=True)
    save_path = str(stats_file)
    with open(save_path, "w") as outfile:
        json.dump(my_dict, outfile, indent=4)
    print(f"Statistics saved for dataset '{dataset_name}' at '{save_path}'")
    print(f"  Actions: {len(actions)} samples, dims: {action_dim}")
    print(f"  States: {len(states)} samples, dims: {state_dim}")
    
    return my_dict



def load_data(hand_states, ee_actions, pose_data_list, arm_states, start_frame, end_frame, use_end_pose, use_delta_pose, relative_to_t0, jump_frame, adaptive_jump_frame, threshold, ee_dim=12):
    """Compute action/state data for a single recording from raw file data.

    Handles all action/state computation logic for one recording.
    Hardware-agnostic; file-format details are in read_vitra_file.
    Called once per recording from load_vitra_data.

    Args:
        hand_states: List of hand state vectors from right_hand_position.txt.
        ee_actions: List of ee command vectors from right_ee.txt.
        pose_data_list: Array of pose dicts from right_arm.npy, or None.
        arm_states: List of joint angle vectors from right_arm.txt, or None.
        start_frame: Episode start frame index, or None.
        end_frame: Episode end frame index, or None.
        use_end_pose: If True, use 6D end pose; otherwise use 7D joint angles.
        use_delta_pose: If True, compute delta poses (only when use_end_pose=True).
        relative_to_t0: If True, compute actions relative to t0 frame.
        jump_frame: Target frame offset for action computation.
        adaptive_jump_frame: Smaller offset used when hand command changes significantly.
        threshold: L1 threshold for switching to adaptive_jump_frame.
        ee_dim: Dimension of end-effector (hand) command vector (default: 12).

    Returns:
        Tuple of (action_data, state_data) lists for this recording.
    """
    action_data = []
    state_data = []

    # Mode 1: Position-based (6D end pose)
    if use_end_pose:
        if pose_data_list is None:
            print(f"Warning: 6D pose file not found (right_arm.npy), skipping")
            return [], []

        # Determine frame range to process
        if start_frame is not None and end_frame is not None:
            # For real_robot with annotations: only use frames within range
            episode_end = min(end_frame, len(pose_data_list) - 1)
            frame_start = start_frame
        else:
            # For VITRA: use all frames
            episode_end = len(pose_data_list) - 1
            frame_start = 0

        # Build per_frame_actions: actions computed at EVERY frame before temporal downsampling.
        # This buffer allows us to: 1) compute adaptive jump frame decisions, 2) accumulate
        # deltas over jump_frame when use_delta_pose=True.
        per_frame_actions = []
        for i in range(len(pose_data_list)):
            # Current 6D pose
            pose_6d_current = [
                pose_data_list[i]['x'],
                pose_data_list[i]['y'],
                pose_data_list[i]['z'],
                pose_data_list[i]['rx'],
                pose_data_list[i]['ry'],
                pose_data_list[i]['rz']
            ]

            # Arm action calculation: pose_t+1 or delta
            if i + 1 < len(pose_data_list):
                next_pose = np.array([
                    pose_data_list[i + 1]['x'],
                    pose_data_list[i + 1]['y'],
                    pose_data_list[i + 1]['z'],
                    pose_data_list[i + 1]['rx'],
                    pose_data_list[i + 1]['ry'],
                    pose_data_list[i + 1]['rz']
                ])

                if use_delta_pose:
                    # FIX: Proper rotation delta calculation using rotation matrices
                    # Translation delta: simple subtraction is correct
                    t_delta = (next_pose[:3] - np.array(pose_6d_current[:3])).tolist()

                    # Rotation delta: R_rel = R_{t+1} * R_t^{-1} (NOT simple Euler subtraction!)
                    # Euler angle subtraction is mathematically incorrect because Euler angles
                    # are non-linear and don't compose additively.
                    R_curr = R.from_euler('xyz', pose_6d_current[3:6], degrees=False)
                    R_next = R.from_euler('xyz', next_pose[3:6], degrees=False)
                    R_rel = R_next * R_curr.inv()  # Proper relative rotation
                    rot_delta = R_rel.as_euler('xyz', degrees=False).tolist()

                    arm_action = t_delta + rot_delta
                else:
                    # Absolute pose: pose_t+1
                    arm_action = next_pose.tolist()
            else:
                # Last frame: use zero delta or current pose
                if use_delta_pose:
                    arm_action = [0.0] * 6
                else:
                    arm_action = pose_6d_current

            # Combine with ee_command
            if i < len(ee_actions):
                per_frame_actions.append(arm_action + ee_actions[i])  # 6 + ee_dim
            else:
                per_frame_actions.append(arm_action + [0.0] * ee_dim)

        # Maximum chunk offsets per t0 (matches training chunk_size)
        max_chunk_offsets = 16

        # Iterate every frame (stride=1) matching reference code.
        # jump_frame is used as the TARGET OFFSET, not as a sampling stride.
        # Each frame independently decides whether to use jump_frame or
        # adaptive_jump_frame based on consecutive ee_command L1 difference.
        for i in range(frame_start, episode_end + 1):
            # Per-frame adaptive check: compare ee_command[i-1] vs ee_command[i]
            # matching reference code's per-frame independence
            current_jump = jump_frame
            if i > 0 and i < len(ee_actions) and (i - 1) < len(ee_actions):
                l1_diff = np.abs(np.array(ee_actions[i]) - np.array(ee_actions[i - 1]))
                max_l1 = np.max(l1_diff)
                if max_l1 > threshold:
                    current_jump = adaptive_jump_frame

            # Collect action and state at current frame
            if i < len(per_frame_actions):
                if use_delta_pose:
                    if relative_to_t0:
                        # Cumulative delta from observation frame (t0=i) to target frames
                        # Matches reference code's relative_to_t0=True approach
                        # Collect multiple chunk offsets per t0 to capture the full
                        # distribution of cumulative deltas seen during training
                        t0_pose = np.array([
                            pose_data_list[i]['x'], pose_data_list[i]['y'], pose_data_list[i]['z'],
                            pose_data_list[i]['rx'], pose_data_list[i]['ry'], pose_data_list[i]['rz']
                        ], dtype=np.float32)
                        R_t0 = R.from_euler('xyz', t0_pose[3:6], degrees=False)

                        for k in range(max_chunk_offsets):
                            target = i + (k + 1) * current_jump
                            if target > episode_end or target >= len(pose_data_list):
                                break
                            target_pose = np.array([
                                pose_data_list[target]['x'], pose_data_list[target]['y'], pose_data_list[target]['z'],
                                pose_data_list[target]['rx'], pose_data_list[target]['ry'], pose_data_list[target]['rz']
                            ], dtype=np.float32)

                            # Translation: direct difference from t0
                            t_delta = (target_pose[:3] - t0_pose[:3]).tolist()

                            # Rotation: R_target * R_t0^{-1}
                            R_target = R.from_euler('xyz', target_pose[3:6], degrees=False)
                            R_rel = R_target * R_t0.inv()
                            rot_delta = R_rel.as_euler('xyz', degrees=False).tolist()

                            # Hand command at target - 1 (matching reference code)
                            hand_idx = max(target - 1, 0)
                            hand_cmd = ee_actions[hand_idx] if hand_idx < len(ee_actions) else [0.0] * ee_dim

                            action = t_delta + rot_delta + hand_cmd
                            action_data.append(action)
                    else:
                        # Frame-to-frame delta: pose[i + jump] - pose[i]
                        target = i + current_jump
                        if target <= episode_end and target < len(pose_data_list):
                            curr_pose = np.array([
                                pose_data_list[i]['x'], pose_data_list[i]['y'], pose_data_list[i]['z'],
                                pose_data_list[i]['rx'], pose_data_list[i]['ry'], pose_data_list[i]['rz']
                            ], dtype=np.float32)
                            target_pose = np.array([
                                pose_data_list[target]['x'], pose_data_list[target]['y'], pose_data_list[target]['z'],
                                pose_data_list[target]['rx'], pose_data_list[target]['ry'], pose_data_list[target]['rz']
                            ], dtype=np.float32)

                            # Translation: delta from current frame to next sampled frame
                            t_delta = (target_pose[:3] - curr_pose[:3]).tolist()

                            # Rotation: R_target * R_curr^{-1}
                            R_curr = R.from_euler('xyz', curr_pose[3:6], degrees=False)
                            R_target = R.from_euler('xyz', target_pose[3:6], degrees=False)
                            R_rel = R_target * R_curr.inv()
                            rot_delta = R_rel.as_euler('xyz', degrees=False).tolist()

                            # Hand command at target - 1 (matching reference code)
                            hand_idx = max(target - 1, 0)
                            hand_cmd = ee_actions[hand_idx] if hand_idx < len(ee_actions) else [0.0] * ee_dim

                            action = t_delta + rot_delta + hand_cmd
                            action_data.append(action)
                else:
                    action_data.append(per_frame_actions[i])

                # State: current 6D pose + hand_status
                pose_6d_state = [
                    pose_data_list[i]['x'],
                    pose_data_list[i]['y'],
                    pose_data_list[i]['z'],
                    pose_data_list[i]['rx'],
                    pose_data_list[i]['ry'],
                    pose_data_list[i]['rz']
                ]
                if i < len(hand_states):
                    state_data.append(pose_6d_state + hand_states[i])

    # Mode 2: Joint-based (7D joint angles)
    else:
        if arm_states is None:
            return [], []

        # Determine frame range to process
        if start_frame is not None and end_frame is not None:
            # For real_robot with annotations: only use frames within range
            episode_end = min(end_frame, len(arm_states) - 1)
            frame_start = start_frame
        else:
            # For VITRA: use all frames
            episode_end = len(arm_states) - 1
            frame_start = 0

        # Build per_frame_actions: actions computed at EVERY frame before temporal downsampling.
        # This buffer allows us to compute adaptive jump frame decisions based on hand command changes.
        per_frame_actions = []
        for i in range(len(arm_states)):
            # Arm action is next arm state (t+1)
            if i + 1 < len(arm_states):
                arm_action = arm_states[i + 1]
            else:
                # Last frame: use current state as action
                arm_action = arm_states[i]

            # Combine with ee_command
            if i < len(ee_actions):
                per_frame_actions.append(arm_action + ee_actions[i])  # 7 + ee_dim
            else:
                per_frame_actions.append(arm_action + [0.0] * ee_dim)

        # Iterate every frame (stride=1) matching reference code.
        # jump_frame is only used as the target offset, not as a sampling stride.
        for i in range(frame_start, episode_end + 1):
            # Per-frame adaptive check: compare ee_command[i-1] vs ee_command[i]
            current_jump = jump_frame
            if i > 0 and i < len(ee_actions) and (i - 1) < len(ee_actions):
                l1_diff = np.abs(np.array(ee_actions[i]) - np.array(ee_actions[i - 1]))
                max_l1 = np.max(l1_diff)
                if max_l1 > threshold:
                    current_jump = adaptive_jump_frame

            # Collect action and state at current frame
            if i < len(per_frame_actions):
                action_data.append(per_frame_actions[i])

                # State: current arm state + hand_status
                if i < len(arm_states) and i < len(hand_states):
                    state_data.append(arm_states[i] + hand_states[i])

    return action_data, state_data



def read_vitra_file(recording_dir):
    """Read raw sensor files from a VITRA/real_robot recording directory.

    Contains all VITRA/real_robot file-format-specific I/O: path conventions and
    data structures (dict fields x/y/z/rx/ry/rz for pose, text files for hand/arm).

    Args:
        recording_dir: Path to the recording directory.

    Returns:
        Tuple of (hand_states, ee_actions, pose_data_list, arm_states).
        Each entry is None if the corresponding file does not exist.
    """
    hand_states = None
    ee_actions = None
    pose_data_list = None
    arm_states = None

    hand_state_file = recording_dir / "hand_status" / "right_hand_position.txt"
    ee_action_file = recording_dir / "ee_command" / "right_ee.txt"

    if hand_state_file.exists():
        with open(hand_state_file, 'r') as f:
            hand_states = [[float(x) for x in line.strip().split()] for line in f]

    if ee_action_file.exists():
        with open(ee_action_file, 'r') as f:
            ee_actions = [[float(x) for x in line.strip().split()] for line in f]

    pose_file = recording_dir / "6d_pose" / "right_arm.npy"
    if pose_file.exists():
        pose_data_list = np.load(pose_file, allow_pickle=True)

    arm_state_file = recording_dir / "arm_status" / "right_arm.txt"
    if arm_state_file.exists():
        with open(arm_state_file, 'r') as f:
            arm_states = [[float(x) for x in line.strip().split()] for line in f]

    return hand_states, ee_actions, pose_data_list, arm_states


def load_vitra_data(root, use_end_pose=False, use_delta_pose=False, relative_to_t0=False, jump_frame=4, adaptive_jump_frame=2, threshold=0.087, extra_roots=None):
    """Load VITRA or real_robot action and state data with adaptive jump frame sampling.

    Discovers recording directories from VITRA (all_metadata_dist.json) or
    real_robot (sensor label folders), iterates over them calling read_vitra_file
    and load_data per recording.

    Args:
        root: Root directory of VITRA or real_robot dataset
        use_end_pose: If True, load 6D end pose instead of 7D joint angles
        use_delta_pose: If True, use delta pose (pose_t+1 - pose_t) instead of absolute pose when use_end_pose=True
        relative_to_t0: If True, compute actions relative to t0 frame; if False, compute frame-to-frame
        jump_frame: Temporal downsampling stride (default: 4)
        adaptive_jump_frame: Smaller stride when hand command changes significantly (default: 2)
        threshold: Threshold for detecting significant hand command changes via L1 norm (default: 0.087)

    Returns:
        Tuple of (action_data, state_data)
    """
    # Check if this is VITRA (has all_metadata_dist.json) or real_robot (has sensor data folders)
    metadata_file = root / "all_metadata_dist.json"

    # Determine dataset type and get list of recording directories
    recording_dirs = []

    if metadata_file.exists():
        # VITRA dataset
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)

        for item in metadata_list:
            relative_path = item['relative_path']
            start_frame = item.get('start', None)
            end_frame = item.get('end', None)
            recording_dirs.append((root / relative_path, relative_path, start_frame, end_frame))

        desc = f"Loading VITRA data ({'6D delta pose' if use_end_pose and use_delta_pose else '6D pose' if use_end_pose else '7D joints'}, jump={jump_frame}, adaptive={adaptive_jump_frame})"
    else:
        # real_robot dataset: browse data folders and load annotations
        sensor_folders = ['acoustic_data', 'radar_data', 'thermal_data']

        for sensor_folder in tqdm(sensor_folders, desc="Scanning sensor folders"):
            sensor_path = root / sensor_folder
            if not sensor_path.exists():
                continue

            label_folder = root / sensor_folder.replace('_data', '_labels')

            # Iterate through episode folders
            episode_list = sorted(sensor_path.iterdir())
            for episode_folder in tqdm(episode_list, desc=f"Discovering {sensor_folder} episodes", leave=False):
                if not episode_folder.is_dir():
                    continue

                episode_name = episode_folder.name
                relative_path = f"{sensor_folder}/{episode_name}"

                # Load label file to get annotations with start/end frames
                label_file = label_folder / f"{episode_name}.json"
                if not label_file.exists():
                    print(f"Warning: Label file not found: {label_file}, skipping episode")
                    continue

                try:
                    with open(label_file, 'r') as f:
                        label_data = json.load(f)

                    # Check if episode is abolished
                    if label_data.get('problem', {}).get('abolished', False):
                        continue

                    # Get annotations
                    annotations = label_data.get('annotations', [])
                    if not annotations:
                        continue

                    # Get total frame count
                    total_frames = label_data.get('frame_num_total', None)

                    # Add each annotation as a separate recording
                    for annotation in annotations:
                        start_frame = annotation.get('start', 0)
                        end_frame = annotation.get('end', total_frames - 1 if total_frames else None)
                        recording_dirs.append((episode_folder, relative_path, start_frame, end_frame))

                except Exception as e:
                    print(f"Warning: Could not read label file {label_file}: {e}")
                    continue

        # Also scan extra_roots if provided
        if extra_roots:
            for extra_root_path in extra_roots:
                extra_root = Path(extra_root_path)
                if not extra_root.exists():
                    print(f"Warning: Extra root not found: {extra_root}")
                    continue

                extra_sensor_folders = ['acoustic_data', 'radar_data', 'thermal_data']
                for sensor_folder in extra_sensor_folders:
                    sensor_path = extra_root / sensor_folder
                    if not sensor_path.exists():
                        continue

                    label_folder = extra_root / sensor_folder.replace('_data', '_labels')

                    episode_list = sorted(sensor_path.iterdir())
                    for episode_folder in episode_list:
                        if not episode_folder.is_dir():
                            continue

                        episode_name = episode_folder.name
                        relative_path = f"{sensor_folder}/{episode_name}"

                        label_file = label_folder / f"{episode_name}.json"
                        if not label_file.exists():
                            print(f"Warning: Label file not found: {label_file}, skipping episode")
                            continue

                        try:
                            with open(label_file, 'r') as f:
                                label_data = json.load(f)

                            if label_data.get('problem', {}).get('abolished', False):
                                continue

                            annotations = label_data.get('annotations', [])
                            if not annotations:
                                continue

                            total_frames = label_data.get('frame_num_total', None)

                            for annotation in annotations:
                                start_frame = annotation.get('start', 0)
                                end_frame = annotation.get('end', total_frames - 1 if total_frames else None)
                                recording_dirs.append((episode_folder, relative_path, start_frame, end_frame))

                        except Exception as e:
                            print(f"Warning: Could not read label file {label_file}: {e}")
                            continue

                print(f"After scanning extra roots: {len(recording_dirs)} total recording segments")

        if not recording_dirs:
            print(f"Warning: No data folders found in {root}")
            return [], []

        print(f"Discovered {len(recording_dirs)} recording segments to process")
        desc = f"Loading real_robot data ({'6D delta pose' if use_end_pose and use_delta_pose else '6D pose' if use_end_pose else '7D joints'}, jump={jump_frame}, adaptive={adaptive_jump_frame})"

    action_data = []
    state_data = []

    total_recordings = len(recording_dirs)
    print(f"\n[Progress] Starting to process {total_recordings} recording segments...", flush=True)

    pbar = tqdm(recording_dirs, desc=desc, mininterval=1.0, file=sys.stdout)
    for idx, recording_info in enumerate(pbar):
        recording_dir, relative_path, start_frame, end_frame = recording_info
        pbar.set_postfix(actions=len(action_data), states=len(state_data), refresh=True)

        # Print progress every 100 recordings
        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"[Progress] Processing {idx + 1}/{total_recordings} - actions: {len(action_data)}, states: {len(state_data)}", flush=True)

        hand_states, ee_actions, pose_data_list, arm_states = read_vitra_file(recording_dir)

        if hand_states is None or ee_actions is None:
            continue

        rec_actions, rec_states = load_data(
            hand_states, ee_actions, pose_data_list, arm_states,
            start_frame, end_frame,
            use_end_pose, use_delta_pose, relative_to_t0,
            jump_frame, adaptive_jump_frame, threshold,
        )  # ee_dim defaults to 12 for VITRA/real_robot
        action_data.extend(rec_actions)
        state_data.extend(rec_states)

    return action_data, state_data

def read_molmoact_file(parquet_file):
    """Read raw data from a MolmoAct parquet file and convert to VITRA-compatible format.

    Follows the same convention as read_vitra_file and read_agibot_file:
      - pose_data_list contains N state poses (current arm state per frame).
      - hand_states contains the current gripper state per frame.
      - ee_actions contains the absolute gripper target per frame.

    State format in parquet: [x, y, z, rx, ry, rz, gripper] (7D)
      - xyz in metres; euler angles in radians; gripper is an absolute position.
    Action format in parquet: [x_target, y_target, z_target, rx_target, ry_target, rz_target, d_gripper] (7D)
      - First 6 dims are the absolute arm target pose (next state).
      - Last dim (d_gripper) is a delta: abs_gripper_target = state_gripper + d_gripper.

    load_data with use_end_pose=True will compute arm_action[t] = pose_data_list[t+1]
    (next state pose), which corresponds to the commanded arm target.

    Args:
        parquet_file: Path to the parquet file.

    Returns:
        hand_states: List of [gripper_state] per frame (N entries).
        ee_actions: List of [abs_gripper_target] per frame (N entries).
        pose_data_list: List of N state pose dicts {x, y, z, rx, ry, rz}.
        arm_states: None (always use end-pose mode for MolmoAct).
        Returns (None, None, None, None) on error.
    """
    try:
        table = pq.read_table(parquet_file)
        N = len(table)
        if N == 0:
            return None, None, None, None

        states_raw = [table['state'][i].as_py() for i in range(N)]
        actions_raw = [table['actions'][i].as_py() for i in range(N)]

        # pose_data_list: N state poses (current arm state per frame).
        # load_data reads arm_action[t] = pose_data_list[t+1] = state pose at t+1.
        pose_data_list = []
        for s in states_raw:
            pose_data_list.append({'x': float(s[0]), 'y': float(s[1]), 'z': float(s[2]),
                                   'rx': float(s[3]), 'ry': float(s[4]), 'rz': float(s[5])})

        # hand_states: current gripper state per frame
        hand_states = [[float(s[6])] for s in states_raw]

        # ee_actions: absolute gripper target = current gripper + delta gripper
        ee_actions = [[float(s[6]) + float(a[6])] for s, a in zip(states_raw, actions_raw)]

        return hand_states, ee_actions, pose_data_list, None

    except Exception as e:
        print(f"Error reading {parquet_file}: {e}")
        return None, None, None, None


def load_molmoact_data(root, dataset_name, use_end_pose=True, use_delta_pose=False, relative_to_t0=False, jump_frame=4, adaptive_jump_frame=2, threshold=0.087):
    """Load MolmoAct action and state data.

    Discovers parquet files from train/data_new/, reads each with
    read_molmoact_file, then calls load_data to extract per-frame
    action and state samples.

    MolmoAct always uses end-pose mode (use_end_pose=True) since the parquet
    files store 6D state poses.  The arm action is pose_data_list[t+1] (next
    state pose).  The gripper action is the accumulated absolute target
    (state_gripper + delta_gripper).

    Args:
        root: Root directory of the MolmoAct dataset.
        dataset_name: Dataset name (e.g. 'molmoact_household', 'molmoact_tabletop').
        use_end_pose: Ignored (always True for MolmoAct); kept for API consistency.
        use_delta_pose: If True, compute delta poses instead of absolute poses.
        relative_to_t0: If True, compute actions relative to t0 frame.
        jump_frame: Target frame offset for action computation (default: 4).
        adaptive_jump_frame: Smaller offset when gripper command changes significantly (default: 2).
        threshold: L1 threshold for switching to adaptive_jump_frame (default: 0.087).

    Returns:
        Tuple of (action_data, state_data).
    """
    split = dataset_name.split('_')[1]  # household or tabletop

    action_data = []
    state_data = []

    data_dir = root / "train" / "data_new"
    if not data_dir.exists():
        print(f"Warning: Data directory not found: {data_dir}")
        return action_data, state_data

    chunk_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    desc = f"Loading MolmoAct {split} data ({'6D delta pose' if use_delta_pose else '6D pose'}, jump={jump_frame}, adaptive={adaptive_jump_frame})"
    pbar = tqdm(chunk_dirs, desc=desc)
    for chunk_dir in pbar:
        pbar.set_postfix(actions=len(action_data), states=len(state_data), refresh=False)
        for parquet_file in sorted(chunk_dir.glob("*.parquet")):
            hand_states, ee_actions, pose_data_list, arm_states = read_molmoact_file(parquet_file)
            if hand_states is None:
                continue

            N = len(hand_states)
            rec_actions, rec_states = load_data(
                hand_states, ee_actions, pose_data_list, arm_states,
                start_frame=0, end_frame=N - 1,
                use_end_pose=True,  # MolmoAct always uses end-pose mode
                use_delta_pose=use_delta_pose,
                relative_to_t0=relative_to_t0,
                jump_frame=jump_frame,
                adaptive_jump_frame=adaptive_jump_frame,
                threshold=threshold,
                ee_dim=1,
            )
            action_data.extend(rec_actions)
            state_data.extend(rec_states)

    return action_data, state_data



def read_agibot_file(h5_file, arm_side):
    """Read raw sensor data from an Agibot HDF5 file and convert to VITRA format.

    Reads end-effector position and orientation (quaternion xyzw) from
    /state/end/position [N,2,3] and /state/end/orientation [N,2,4], plus
    effector (hand) commands from /action/effector/position.

    Converts quaternion orientation to Euler angles (xyz, radians) to match the
    VITRA pose_data_list format: list of dicts with keys 'x','y','z','rx','ry','rz'.

    Args:
        h5_file: Path to the HDF5 file.
        arm_side: 'left' (index 0) or 'right' (index 1).

    Returns:
        Tuple (hand_states, ee_actions, pose_data_list, arm_states).
        hand_states: list of per-frame effector command vectors (floats).
        ee_actions: same as hand_states (used as action command in load_data).
        pose_data_list: list of dicts {'x','y','z','rx','ry','rz'} from /action/end/*.
        arm_states: None (not used for end-pose mode).
        effector_dim: int, dimension of effector per hand (1 for gripper, 6 for agihand).
        Returns (None, None, None, None, None) on error.
    """
    side_idx = 0 if arm_side == 'left' else 1

    try:
        with h5py.File(h5_file, 'r') as hf:
            # End pose: positions [N,2,3] and orientations [N,2,4] (xyzw quaternion)
            action_pos = hf['/action/end/position'][:]       # [N, 2, 3]
            action_ori = hf['/action/end/orientation'][:]    # [N, 2, 4]  xyzw
            state_pos  = hf['/state/end/position'][:]        # [N, 2, 3]
            state_ori  = hf['/state/end/orientation'][:]     # [N, 2, 4]  xyzw

            # Effector (hand) commands: [N, 2] gripper or [N, 12] agihand
            action_eff = hf['/action/effector/position'][:]  # [N, effector_total]
            state_eff  = hf['/state/effector/position'][:]   # [N, effector_total]

        N = state_pos.shape[0]
        effector_total = action_eff.shape[1]
        effector_per_hand = effector_total // 2  # 1 for gripper, 6 for agihand

        # Effector slice for this arm
        if arm_side == 'left':
            eff_slice = slice(0, effector_per_hand)
        else:
            eff_slice = slice(effector_per_hand, effector_total)

        # Build pose_data_list from ACTION end pose (matches VITRA convention: action contains next pose)
        pose_data_list = []
        for i in range(N):
            xyz = state_pos[i, side_idx]          # [3]  meters
            quat_xyzw = state_ori[i, side_idx]    # [4]  xyzw
            # Convert quaternion (xyzw) to Euler xyz (radians)
            rot = R.from_quat(quat_xyzw)           # scipy expects xyzw
            euler_xyz = rot.as_euler('xyz', degrees=False)
            pose_data_list.append({
                'x': float(xyz[0]),
                'y': float(xyz[1]),
                'z': float(xyz[2]),
                'rx': float(euler_xyz[0]),
                'ry': float(euler_xyz[1]),
                'rz': float(euler_xyz[2]),
            })

        # Build hand_states from STATE effector (per-frame current hand state)
        hand_states = [state_eff[i, eff_slice].tolist() for i in range(N)]

        # ee_actions = action effector commands (used as the "ee_command" in load_data)
        ee_actions = [action_eff[i, eff_slice].tolist() for i in range(N)]

        # arm_states is None: we always use end-pose mode for Agibot
        arm_states = None

        return hand_states, ee_actions, pose_data_list, arm_states, effector_per_hand

    except Exception as e:
        print(f"Error reading {h5_file}: {e}")
        return None, None, None, None, None


def load_agibot_data(root, use_end_pose=True, use_delta_pose=False, relative_to_t0=False, jump_frame=4, adaptive_jump_frame=2, threshold=0.087):
    """Load Agibot action and state data, separated by gripper and agihand.

    Discovers episodes from extra_sensor.jsonl, reads start/end frames from
    task_<task_id>.json, then calls read_agibot_file + load_data for each arm
    side independently. Results are separated into gripper and agihand buckets.

    Args:
        root: Root directory of the Agibot dataset.
        use_end_pose: If True, use 6D end pose (always True for Agibot).
        use_delta_pose: If True, use delta pose instead of absolute pose.
        relative_to_t0: If True, compute actions relative to t0 frame.
        jump_frame: Target frame offset for action computation (default: 4).
        adaptive_jump_frame: Smaller offset when hand command changes significantly (default: 2).
        threshold: L1 threshold for switching to adaptive_jump_frame (default: 0.087).

    Returns:
        Tuple of (action_data_gripper, state_data_gripper,
                  action_data_agihand, state_data_agihand).
    """
    action_data_gripper = []
    state_data_gripper = []
    action_data_agihand = []
    state_data_agihand = []

    metadata_file = root / "task_info" / "extra_sensor.jsonl"
    if not metadata_file.exists():
        print(f"Warning: Metadata file not found: {metadata_file}")
        return action_data_gripper, state_data_gripper, action_data_agihand, state_data_agihand

    # Cache for task JSON files: task_id -> {episode_id: (start_frame, end_frame)}
    task_frame_cache = {}

    def get_episode_frames(task_id, episode_id):
        """Return (start_frame, end_frame) from task_<task_id>.json, or (None, None)."""
        if task_id not in task_frame_cache:
            task_json = root / "task_info" / f"task_{task_id}.json"
            cache = {}
            if task_json.exists():
                try:
                    with open(task_json, 'r', encoding='utf-8') as tf:
                        task_data = json.load(tf)
                    # task_data may be a list of episode dicts or a single dict
                    episodes = task_data if isinstance(task_data, list) else [task_data]
                    for ep in episodes:
                        eid = str(ep.get('episode_id', ''))
                        action_configs = ep.get('label_info', {}).get('action_config', [])
                        if action_configs:
                            starts = [ac['start_frame'] for ac in action_configs if 'start_frame' in ac]
                            ends   = [ac['end_frame']   for ac in action_configs if 'end_frame'   in ac]
                            if starts and ends:
                                cache[eid] = (min(starts), max(ends))
                except Exception as e:
                    print(f"Warning: Could not read {task_json}: {e}")
            task_frame_cache[task_id] = cache
        return task_frame_cache[task_id].get(str(episode_id), (None, None))

    # Collect recording_dirs entries: (h5_file, arm_side, start_frame, end_frame)
    # We process left and right arms separately so load_data handles each as an independent recording.
    recording_dirs_gripper = []
    recording_dirs_agihand = []

    with open(metadata_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    scan_pbar = tqdm(lines, desc="Scanning Agibot episodes")
    for line in scan_pbar:
        item = json.loads(line)
        task_id   = item.get('task_id', '')
        episode_id = item.get('episode_id', '')
        if not task_id or not episode_id:
            continue

        h5_file = root / "proprio_stats" / str(task_id) / str(episode_id) / "proprio_stats.h5"
        if not h5_file.exists():
            continue

        # Peek effector dimension to bucket into gripper vs agihand
        try:
            with h5py.File(h5_file, 'r') as hf:
                effector_total = hf['/action/effector/position'].shape[1]
        except Exception as e:
            print(f"Warning: Cannot read effector dim from {h5_file}: {e}")
            continue

        start_frame, end_frame = get_episode_frames(task_id, episode_id)

        entry = (h5_file, start_frame, end_frame)
        if effector_total == 2:    # gripper
            recording_dirs_gripper.append(entry)
        elif effector_total == 12:  # agihand
            recording_dirs_agihand.append(entry)

        scan_pbar.set_postfix(gripper=len(recording_dirs_gripper), agihand=len(recording_dirs_agihand), refresh=False)

    def process_recordings(recording_dirs, label):
        """Run read_agibot_file + load_data for all entries, combining both arms per frame.

        Each output sample is [left_features | right_features] concatenated so that
        compute_stats_from_data can split in half to obtain per-arm statistics.
        """
        all_action = []
        all_state  = []
        pbar = tqdm(recording_dirs, desc=f"Loading Agibot {label} ({'6D delta pose' if use_delta_pose else '6D pose'}, jump={jump_frame}, adaptive={adaptive_jump_frame})")
        for h5_file, start_frame, end_frame in pbar:
            per_arm = {}
            for arm_side in ('left', 'right'):
                pbar.set_postfix(arm=arm_side, actions=len(all_action), states=len(all_state), refresh=True)
                hand_states, ee_actions, pose_data_list, arm_states, effector_per_hand = \
                    read_agibot_file(h5_file, arm_side)
                if hand_states is None:
                    break

                rec_actions, rec_states = load_data(
                    hand_states, ee_actions, pose_data_list, arm_states,
                    start_frame, end_frame,
                    use_end_pose=use_end_pose,
                    use_delta_pose=use_delta_pose,
                    relative_to_t0=relative_to_t0,
                    jump_frame=jump_frame,
                    adaptive_jump_frame=adaptive_jump_frame,
                    threshold=threshold,
                    ee_dim=effector_per_hand,
                )
                per_arm[arm_side] = (rec_actions, rec_states)

            if 'left' not in per_arm or 'right' not in per_arm:
                continue

            left_actions, left_states = per_arm['left']
            right_actions, right_states = per_arm['right']

            # Zip frame-by-frame: each combined sample = [left | right]
            for la, ra in zip(left_actions, right_actions):
                all_action.append(la + ra)
            for ls, rs in zip(left_states, right_states):
                all_state.append(ls + rs)

        return all_action, all_state

    if recording_dirs_gripper:
        action_data_gripper, state_data_gripper = process_recordings(recording_dirs_gripper, 'gripper')

    if recording_dirs_agihand:
        action_data_agihand, state_data_agihand = process_recordings(recording_dirs_agihand, 'agihand')

    return action_data_gripper, state_data_gripper, action_data_agihand, state_data_agihand

if __name__=='__main__':

    # Define dataset paths
    # Production: Muse_Robot_Dataset_01 and _02 in main-blob
    root_dir = '<PROJECT_PATH>'
    datasets = {
        'real_robot_merged': '<PROJECT_PATH>',
    }

    extra_roots_map = {
        'real_robot_merged': ['<PROJECT_PATH>'],
    }
    
    action_type = 'angle'
    use_end_pose = True
    use_delta_pose = True
    relative_to_t0 = False
    jump_frame = 4
    adaptive_jump_frame = 2
    threshold = 0.087
    
    # Check mode from command line arguments
    
    mode = 'statistics'
    
    if mode == 'filter_videos':
        print("\n" + "="*60)
        print("FILTERING VALID VIDEOS FOR ALL DATASETS")
        print("="*60)
        
        # Filter and cache valid videos for all datasets
        for dataset_name, dataset_path in datasets.items():
            try:
                filter_and_cache_valid_videos(dataset_name, dataset_path)
            except Exception as e:
                print(f"Error filtering videos for {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    elif mode == 'frame_counts':
        print("\n" + "="*60)
        print("COMPUTING FRAME COUNTS FOR ALL DATASETS")
        print("="*60)
        
        # Compute frame counts for all datasets
        for dataset_name, dataset_path in datasets.items():
            try:
                compute_frame_counts(dataset_name, dataset_path)
            except Exception as e:
                print(f"Error computing frame counts for {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    elif mode == 'statistics':
        print("\n" + "="*60)
        print("COMPUTING STATISTICS FOR ALL DATASETS")
        print("="*60)
        
        # Process each dataset for statistics
        for dataset_name, dataset_path in datasets.items():
            print(f"\n{'='*60}")
            print(f"Calculating statistics for dataset: {dataset_name}")
            print(f"{'='*60}")
            try:
                # Compute statistics with joint-based mode (default)
                # compute_statistics_robot(dataset_name, dataset_path, action_type, use_end_pose=False, 
                #                         jump_frame=jump_frame, adaptive_jump_frame=adaptive_jump_frame, threshold=threshold)
                
                # For VITRA and real_robot, also compute statistics with position-based mode
                print(f"\n{'='*60}")
                print(f"Calculating 6D pose statistics for dataset: {dataset_name}")
                print(f"{'='*60}")
                extra_roots = extra_roots_map.get(dataset_name, [])
                compute_statistics_robot(dataset_name, dataset_path, action_type, use_end_pose=use_end_pose, use_delta_pose=use_delta_pose, relative_to_t0=relative_to_t0,
                                        jump_frame=jump_frame, adaptive_jump_frame=adaptive_jump_frame, threshold=threshold, extra_roots=extra_roots)
            except Exception as e:
                print(f"Error processing {dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python calculate_statistics.py [filter_videos|frame_counts|statistics]")
        print("  filter_videos: Validate videos with ffmpeg and generate valid_videos.json")
        print("  frame_counts: Generate frame count cache files")
        print("  statistics: Compute action/state statistics (default)")