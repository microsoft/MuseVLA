"""
robot_dataset_vlm_vla.py

Dataset implementation for VLM-VLA co-training on robot datasets.
Combines VLM sensor token prediction with VLA action prediction.

Note: This dataset uses lazy initialization and cached frame counts for fast startup.
"""

import json
import logging
import os
import random
import signal
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R  # For proper rotation delta calculation

logger = logging.getLogger(__name__)

# Default timeout for video loading operations (in seconds)
VIDEO_LOAD_TIMEOUT = 15  # Keep low to avoid blocking NCCL collectives on other ranks

# Maximum wall-clock seconds for the entire retry loop in __getitem__.
# Must be well below NCCL timeout (~10 min default) so we return a dummy
# instead of stalling the distributed training group.
MAX_RETRY_WALL_SECONDS = 120

# Thread pool for video loading with timeout
_video_load_executor = ThreadPoolExecutor(max_workers=4)
_video_load_consecutive_timeouts = 0  # Track consecutive timeouts to detect pool exhaustion


def _load_frame_from_video_impl(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """Internal implementation of frame loading from video.

    This function is called in a separate thread to allow timeout.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0 or frame_idx >= frame_count:
            cap.release()
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return None

        return frame
    except Exception as e:
        logger.warning(f"Exception loading frame from {video_path}: {e}")
        return None


def _maybe_reset_video_pool() -> None:
    """Recreate the thread pool if too many consecutive timeouts indicate zombie threads."""
    global _video_load_executor, _video_load_consecutive_timeouts
    if _video_load_consecutive_timeouts >= 4:
        logger.warning("Resetting video load thread pool after %d consecutive timeouts (likely zombie threads)",
                       _video_load_consecutive_timeouts)
        # Don't shutdown old pool (threads are stuck in C code and won't join);
        # just orphan it and create a fresh one.
        _video_load_executor = ThreadPoolExecutor(max_workers=4)
        _video_load_consecutive_timeouts = 0


def load_frame_with_timeout(video_path: str, frame_idx: int, timeout: float = VIDEO_LOAD_TIMEOUT) -> Optional[np.ndarray]:
    """Load a frame from video with timeout protection.

    Args:
        video_path: Path to the video file
        frame_idx: Frame index to load
        timeout: Maximum time to wait (seconds)

    Returns:
        BGR frame as numpy array, or None if loading fails/times out
    """
    global _video_load_consecutive_timeouts
    _maybe_reset_video_pool()
    try:
        future = _video_load_executor.submit(_load_frame_from_video_impl, video_path, frame_idx)
        result = future.result(timeout=timeout)
        if result is not None:
            _video_load_consecutive_timeouts = 0
        else:
            _video_load_consecutive_timeouts += 1
        return result
    except FuturesTimeoutError:
        _video_load_consecutive_timeouts += 1
        logger.warning(f"Timeout ({timeout}s) loading frame {frame_idx} from {video_path}")
        return None
    except Exception as e:
        _video_load_consecutive_timeouts += 1
        logger.warning(f"Error loading frame from {video_path}: {e}")
        return None


def get_video_frame_count_with_timeout(video_path: str, timeout: float = VIDEO_LOAD_TIMEOUT) -> int:
    """Get video frame count with timeout protection.

    Args:
        video_path: Path to the video file
        timeout: Maximum time to wait (seconds)

    Returns:
        Frame count, or 0 if failed/timed out
    """
    global _video_load_consecutive_timeouts

    def _get_frame_count():
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return count
        except Exception:
            return 0

    _maybe_reset_video_pool()
    try:
        future = _video_load_executor.submit(_get_frame_count)
        result = future.result(timeout=timeout)
        if result > 0:
            _video_load_consecutive_timeouts = 0
        return result
    except FuturesTimeoutError:
        _video_load_consecutive_timeouts += 1
        logger.warning(f"Timeout ({timeout}s) getting frame count from {video_path}")
        return 0
    except Exception as e:
        _video_load_consecutive_timeouts += 1
        logger.warning(f"Error getting frame count from {video_path}: {e}")
        return 0

from musevla.datasets.dataset_utils import ActionFeature, StateFeature
from musevla.datasets.human_dataset import pad_action
from musevla.datasets.robot_dataset import (
    pad_state_robot,
    transfer_xhand_to_human,
    transfer_gripper_to_human,
    transfer_agihand_to_human,
)
from musevla.utils.data_utils import GaussianNormalizer, read_dataset_statistics

# Instruction → (sensor_token, primary_object) mapping for real robot datasets.
# ⚠️ REVIEW THESE MAPPINGS — verify sensor/object assignments are correct.
INSTRUCTION_SENSOR_MAP = {
    # Dataset_01 instructions (11 types)
    "Pick up the towel covering the ringing phone.": ("audio", "towels"),
    "Place the towel into the basket.": ("none", "towels"),
    "Pick up the clothes covering the ringing phone.": ("audio", "clothes"),
    "Place the clothes into the basket.": ("none", "clothes"),
    "Pick up the room-temperature drink.": ("thermal", "bottles"),
    "Pick up the cold drink.": ("thermal", "bottles"),
    "Pick up the hot drink.": ("thermal", "bottles"),
    "Place the drink into the basket.": ("none", "bottles"),
    "Open the occupied box.": ("radar", "boxes"),
    "Pick up the item in the box.": ("none", "boxes"),
    "Place the item into the basket.": ("none", "boxes"),
    # Dataset_02 instructions (4 types, "Place the drink..." already covered above)
    "Pick up the hot drink in the box.": ("thermal", "bottles"),
    "Pick up the cold drink in the box.": ("thermal", "bottles"),
    "Pick up the room-temperature drink in the box.": ("thermal", "bottles"),
}


class RoboDatasetVLMVLACore(Dataset):
    """
    Dataset class for VLM-VLA co-training on robot manipulation data.
    
    Loads RGB images + sensor overlay images, instructions, metadata,
    actions, and states for combined sensor prediction and action prediction tasks.
    
    Supports four robot datasets:
    - VITRA: all_metadata.json with extra_sensor.json
    - Real Robot (Muse): radar/acoustic/thermal label JSON files
    - MolmoAct: meta/extra_sensor.json
    - AgiBotWorld: task_info/extra_sensor.jsonl
    """
    
    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        statistics_path: str,
        sensor_token_prob: Dict[str, float],
        background_modes: str,
        placeholder_replacements_path: Optional[str] = None,
        placeholder_text: str = "<placeholder>",
        processor=None,
        action_past_window_size: int = 0,
        action_future_window_size: int = 16,
        image_past_window_size: int = 0,
        image_future_window_size: int = 0,
        seed: int = 42,
        max_samples: Optional[int] = None,
        action_type: str = "angle",
        max_retries: int = 10,
        random_sample_sensor: bool = True,
        sensor_masked: bool = False,
        data_mix_sensor: str = "acoustic, radar, thermal",
        jump_frame: int = 1,
        adaptive_jump_frame: int = 2,
        threshold: float = 0.087,
        sensor_input_all: bool = False,
        RGB_only: bool = False,
        use_end_pose: bool = False,
        use_delta_pose: bool = False,
        relative_to_t0: bool = False,
        random_drop_RGB: float = 0,
        radar_fix_start_frame: bool = False,
        use_non_vitra_ee_action: bool = True,
        data_fraction: float = 1.0,
        filter_invalid_videos: bool = True,
        train_seg_token: bool = False,
        text_generalization_prob: float = 0.0,
        visual_clone_prob: float = 0.0,
        extra_roots: Optional[List[str]] = None,
        use_validity_filter: bool = True,
    ):
        """
        Initialize VLM-VLA co-training dataset.
        
        Args:
            dataset_name: Name of dataset (vitra/real_robot/molmoact_household/molmoact_tabletop/agibot)
            root_dir: Root directory containing the dataset
            statistics_path: Path to normalization statistics JSON file
            sensor_token_prob: Probability dict for sampling sensor tokens
            background_modes: String of background modes (dark/color/grey) for sensor overlays
            placeholder_replacements_path: Path to folder containing word probability JSON files
            placeholder_text: Placeholder string in instructions
            processor: Text/image processor with tokenizer
            action_past_window_size: Number of past action frames to include
            action_future_window_size: Number of future action frames to predict (chunk_size)
            image_past_window_size: Number of past image frames to include
            image_future_window_size: Number of future image frames to include
            seed: Random seed for reproducibility
            max_samples: Maximum number of samples to load (None for all)
            action_type: Type of action representation ('angle' or 'keypoints')
            max_retries: Maximum number of retries if sample loading fails
            jump_frame: Temporal downsampling stride for action sequences (1=no downsampling)
            adaptive_jump_frame: Smaller temporal stride when hand state changes significantly
            threshold: Threshold for detecting significant hand command changes (L1 norm)
            use_end_pose: Whether to use 6D end pose (True) or 7D joint angles (False) for arm actions
            use_delta_pose: Whether to use delta pose (pose_t+1 - pose_t) instead of absolute pose when use_end_pose=True
            relative_to_t0: Whether to compute actions relative to t0 frame (True) or frame-to-frame (False)
            random_drop_RGB: Probability of randomly dropping RGB tensor (setting to zeros) when sensor is present
        """
        self.dataset_name = dataset_name
        self.root = Path(root_dir)
        self.statistics_path = statistics_path
        self.sensor_token_prob = sensor_token_prob
        self.background_modes = background_modes
        self.placeholder_replacements_path = placeholder_replacements_path
        self.placeholder_text = placeholder_text
        self.processor = processor
        self.action_past_window_size = action_past_window_size
        self.action_future_window_size = action_future_window_size
        self.chunk_size = action_future_window_size
        self.image_past_window_size = image_past_window_size
        self.image_future_window_size = image_future_window_size
        self.seed = seed
        self.max_samples = max_samples
        self.action_type = action_type
        self.max_retries = max_retries
        self.random_sample_sensor = random_sample_sensor
        self.sensor_masked = sensor_masked
        self.jump_frame = jump_frame
        self.adaptive_jump_frame = adaptive_jump_frame
        self.threshold = threshold
        self.sensor_input_all = sensor_input_all
        self.use_end_pose = use_end_pose
        self.use_delta_pose = use_delta_pose
        self.relative_to_t0 = relative_to_t0
        self.RGB_only = RGB_only
        self.random_drop_RGB = random_drop_RGB
        self.radar_fix_start_frame = radar_fix_start_frame
        self.use_non_vitra_ee_action = use_non_vitra_ee_action
        self.data_fraction = data_fraction
        self.filter_invalid_videos = filter_invalid_videos
        self.train_seg_token = train_seg_token
        self.text_generalization_prob = text_generalization_prob
        self.visual_clone_prob = visual_clone_prob
        self.extra_roots = [Path(p) for p in extra_roots] if extra_roots else []
        self.use_validity_filter = use_validity_filter
        
        # Parse data_mix into list of sensors
        self.data_mix_sensors = [s.strip() for s in data_mix_sensor.split(',')]
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Normalize sensor token probabilities
        total_prob = sum(self.sensor_token_prob.values())
        self.sensor_token_prob = {k: v/total_prob for k, v in self.sensor_token_prob.items()}
        
        # Load word replacement probabilities
        self.word_probs = self._load_word_probabilities()
        
        # Load normalization statistics
        # For agibot, load both gripper and agihand statistics
        if dataset_name == 'agibot':
            # Build mode suffix matching compute_stats_from_data convention
            agibot_mode_suffix = ''
            if use_end_pose:
                if use_delta_pose:
                    agibot_mode_suffix += '_6dpose_delta'
                else:
                    agibot_mode_suffix += '_6dpose'
            if not relative_to_t0:
                agibot_mode_suffix += '_NRt0'

            stats_dir = Path(statistics_path).parent
            print(f"[INIT] agibot use_end_pose={use_end_pose}, use_delta_pose={use_delta_pose}, relative_to_t0={relative_to_t0}")
            print(f"[INIT] agibot mode suffix: '{agibot_mode_suffix}'")

            # Load gripper statistics
            statistics_path_gripper = str(stats_dir / f"statistics_gripper{agibot_mode_suffix}.json")
            print(f"[INIT] Looking for gripper statistics at: {statistics_path_gripper}")
            if Path(statistics_path_gripper).exists():
                self.data_statistics_gripper = read_dataset_statistics(statistics_path_gripper)
                self.gaussian_normalizer_gripper = GaussianNormalizer(
                    self.data_statistics_gripper
                )
                print(f"[INIT] ✓ Loaded gripper statistics from: {statistics_path_gripper}")
            else:
                print(f"[INIT] ✗ WARNING: Gripper statistics not found at {statistics_path_gripper}")
                self.gaussian_normalizer_gripper = None

            # Load agihand statistics
            statistics_path_agihand = str(stats_dir / f"statistics_agihand{agibot_mode_suffix}.json")
            print(f"[INIT] Looking for agihand statistics at: {statistics_path_agihand}")
            if Path(statistics_path_agihand).exists():
                self.data_statistics_agihand = read_dataset_statistics(statistics_path_agihand)
                self.gaussian_normalizer_agihand = GaussianNormalizer(
                    self.data_statistics_agihand
                )
                print(f"[INIT] ✓ Loaded agihand statistics from: {statistics_path_agihand}")
            else:
                print(f"[INIT] ✗ WARNING: AgiHand statistics not found at {statistics_path_agihand}")
                self.gaussian_normalizer_agihand = None

            # Set default normalizer (will be selected per sample based on effector type)
            self.gaussian_normalizer = self.gaussian_normalizer_gripper
        else:
            # For VITRA/real_robot, load appropriate statistics based on pose mode
            if (dataset_name in ['vitra', 'real_robot', 'real_robot_old', 'real_robot_merged', 'real_robot_merged_test', 'muse_robot', 'molmoact_tabletop', 'molmoact_household', 'hand_pretrain'] and use_end_pose):
                # Choose statistics file based on pose mode and relative_to_t0
                # For real_robot_merged, use same statistics as real_robot (no _merged prefix)
                # Only real_robot_merged_test uses _merged prefix for local testing
                merged_prefix = '_merged' if dataset_name == 'real_robot_merged_test' else ''
                if use_delta_pose:
                    base_stats_name = f'statistics{merged_prefix}_6dpose_delta'
                else:
                    base_stats_name = f'statistics{merged_prefix}_6dpose'
                if not relative_to_t0:
                    base_stats_name += '_NRt0'
                statistics_path_pose = str(Path(statistics_path).parent / f'{base_stats_name}.json')
                
                print(f"[INIT] use_end_pose={use_end_pose}, use_delta_pose={use_delta_pose}, relative_to_t0={relative_to_t0}")
                print(f"[INIT] Looking for statistics at: {statistics_path_pose}")
                print(f"[INIT] File exists: {Path(statistics_path_pose).exists()}")
                
                if Path(statistics_path_pose).exists():
                    self.data_statistics = read_dataset_statistics(statistics_path_pose)
                    print(f"[INIT] ✓ Loaded statistics from: {statistics_path_pose}")
                    print(f"[INIT] action_right_std[:6] = {self.data_statistics['action_right_std'][:6]}")
                else:
                    print(f"[INIT] ✗ WARNING: 6D pose statistics not found at {statistics_path_pose}, falling back to {statistics_path}")
                    self.data_statistics = read_dataset_statistics(statistics_path)
            else:
                self.data_statistics = read_dataset_statistics(statistics_path)
            self.gaussian_normalizer = GaussianNormalizer(
                self.data_statistics 
            )
        
        # Load cached frame counts (optional — used only for agibot)
        frame_count_cache_path = self.root / f"{dataset_name}_frame_counts.json"
        if frame_count_cache_path.exists():
            with open(frame_count_cache_path, 'r') as f:
                self.frame_counts = json.load(f)
            # logger.info(f"Loaded frame count cache with {len(self.frame_counts)} videos")
        else:
            # For vitra/real_robot, frame range comes from metadata start/end fields
            self.frame_counts = {}
        
        # logger.info(f"Loaded frame count cache with {len(self.frame_counts)} videos")
        
        # Load invalid videos cache (optional but recommended)
        invalid_videos_cache_path = self.root / f"{dataset_name}_invalid_videos.json"
        if self.filter_invalid_videos and invalid_videos_cache_path.exists():
            with open(invalid_videos_cache_path, 'r') as f:
                self.invalid_videos = set(json.load(f))
            logger.info(f"Loaded invalid videos cache with {len(self.invalid_videos)} filtered videos")
        else:
            if not self.filter_invalid_videos:
                logger.info(f"Skipping invalid videos filter for {dataset_name} (filter_invalid_videos=False)")
            self.invalid_videos = None
        
        # Parse dataset structure and load metadata
        self.samples = self._load_samples(self.max_samples)
        
        # Build frame index: map dataset index to (sample_idx, frame_idx)
        # Also build reverse mapping for efficient video-level jumping
        self.frame_indices = []
        self.sample_to_frame_range = {}  # {sample_idx: (start_idx, end_idx)}
        
        for sample_idx, sample in enumerate(self.samples):
            start_idx = len(self.frame_indices)
            
            # For vitra, real_robot, real_robot_old, and hand_pretrain, use annotation range (start_frame/end_frame)
            if sample.get('dataset') in ('vitra', 'real_robot', 'real_robot_old', 'hand_pretrain'):
                start_frame = sample.get('start_frame', 0)
                end_frame = sample.get('end_frame', sample['num_frames'] - 1)
                valid_length = end_frame - start_frame + 1
                max_start_idx = max(0, valid_length - self.chunk_size)
            else:
                # Each sample can have multiple valid starting frames
                max_start_idx = max(0, sample['num_frames'] - self.chunk_size)
            
            for frame_idx in range(max_start_idx + 1):
                self.frame_indices.append((sample_idx, frame_idx))
            
            end_idx = len(self.frame_indices)
            self.sample_to_frame_range[sample_idx] = (start_idx, end_idx)
        
        logger.info(f"Loaded {len(self.samples)} samples from {dataset_name}")
        logger.info(f"Total frames: {len(self.frame_indices)}")
    
    def _load_word_probabilities(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Load word replacement probability distributions from JSON files.
        
        Returns:
            Dictionary mapping sensor type -> category (semantic label) -> {word: probability}
            e.g., {'thermal': {'cold': {'chilly': 0.5, 'freezing': 0.3}, 'hot': {...}}, ...}
        """
        if self.placeholder_replacements_path is None:
            return {}
        
        word_probs = {}
        replacements_path = Path(self.placeholder_replacements_path)
        
        if not replacements_path.exists():
            logger.warning(f"Word replacements path {replacements_path} does not exist")
            return {}
        
        # Map sensor token names to file prefixes
        sensor_file_map = {
            'thermal': 'thermal_word_probs_en.json',
            'audio': 'acoustic_word_probs_en.json',
            'radar': 'mmwave_word_probs_en.json',
        }
        
        for sensor_type, filename in sensor_file_map.items():
            file_path = replacements_path / filename
            if not file_path.exists():
                logger.warning(f"Word prob file not found: {file_path}")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Preserve category structure
                # Expected structure: {sensor_key: {category_name: {word: prob}}}
                sensor_categories = {}
                for sensor_key, categories in data.items():
                    # categories is a dict of category_name -> {word: prob}
                    for category_name, word_dict in categories.items():
                        # Normalize probabilities for this category
                        total_prob = sum(word_dict.values())
                        if total_prob > 0:
                            normalized_words = {word: prob/total_prob for word, prob in word_dict.items()}
                            sensor_categories[category_name] = normalized_words
                
                word_probs[sensor_type] = sensor_categories
                # logger.info(f"Loaded {len(sensor_categories)} categories for {sensor_type} sensor")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        return word_probs
    
    def _load_samples(self, max_samples: Optional[int] = None) -> List[Dict]:
        """Load samples based on dataset type."""
        if self.dataset_name == 'vitra':
            samples = self._load_vitra_samples(max_samples)
        elif self.dataset_name in ('real_robot', 'real_robot_old'):
            samples = self._load_real_robot_samples(max_samples)
        elif self.dataset_name == 'real_robot_merged':
            samples = self._load_real_robot_merged_samples(max_samples)
        elif self.dataset_name == 'real_robot_merged_test':
            samples = self._load_real_robot_merged_samples(max_samples)
        elif self.dataset_name == 'muse_robot':
            samples = self._load_real_robot_merged_samples(max_samples)
        elif 'molmoact' in self.dataset_name:
            samples = self._load_molmoact_samples(max_samples)
        elif self.dataset_name == 'agibot':
            samples = self._load_agibot_samples(max_samples)
        elif self.dataset_name == 'hand_pretrain':
            samples = self._load_hand_pretrain_samples(max_samples)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        if self.data_fraction < 1.0:
            total = len(samples)
            stride = max(1, int(round(1.0 / self.data_fraction)))
            samples = samples[::stride]
            logger.info(f"data_fraction={self.data_fraction}: keeping {len(samples)}/{total} samples (stride={stride})")

        return samples
    
    def _load_vitra_samples(self, max_samples: Optional[int] = None) -> List[Dict]:
        """Load VITRA dataset samples."""
        metadata_file = self.root / "all_metadata_dist.json"
        if not metadata_file.exists():
            logger.warning(f"Metadata file not found: {metadata_file}")
            return []
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)
        
        samples = []
        for item in metadata_list:
            # Early termination if max_samples reached
            if max_samples is not None and len(samples) >= max_samples:
                break
            
            # Check invalid list FIRST (in-memory) before any filesystem access
            video_key = str(Path(item['relative_path']) / "video.mp4")
            if self.invalid_videos is not None and video_key in self.invalid_videos:
                continue
                
            video_path = self.root / item['relative_path'] / "video.mp4"
            
            rephrased = item.get('rephrased_instruction', '')
            if not rephrased:
                rephrased = item.get('instruction', '')
            
            # Use start/end from metadata (same as real_robot)
            start_frame = item.get('start', None)
            end_frame = item.get('end', None)
            
            if start_frame is not None and end_frame is not None:
                num_frames = end_frame - start_frame + 1
            else:
                # Fallback to frame_counts cache if start/end not present
                video_key = str(Path(item['relative_path']) / "video.mp4")
                if video_key not in self.frame_counts:
                    logger.warning(f"Frame count not found and no start/end for {video_key}")
                    continue
                num_frames = self.frame_counts[video_key]
            
            samples.append({
                'video_path': str(video_path),
                'instruction': rephrased,
                'primary_object': item.get('primary_object', ''),
                'all_objects': item.get('all_objects', ''),
                'dataset': 'vitra',
                'relative_path': item['relative_path'],
                'num_frames': num_frames,
                'start_frame': start_frame if start_frame is not None else 0,
                'end_frame': end_frame if end_frame is not None else num_frames - 1,
            })
        
        return samples
    
    def _load_hand_pretrain_samples(self, max_samples: Optional[int] = None) -> List[Dict]:
        """Load hand pretrain dataset samples (VITRA-style directory layout).

        Expects:
          - {root}/all_metadata_dist.json (or all_metadata.json as fallback)
          - {root}/{episode_id}/video.mp4
          - {root}/{episode_id}/6d_pose/{side}_arm.npy
          - {root}/{episode_id}/hand_status/{side}_hand_position.txt
          - {root}/{episode_id}/ee_command/{side}_ee.txt
        """
        metadata_file = self.root / "all_metadata_dist.json"
        if not metadata_file.exists():
            metadata_file = self.root / "all_metadata.json"
        if not metadata_file.exists():
            logger.warning(f"Hand pretrain metadata not found: {metadata_file}")
            return []

        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)

        samples = []
        for item in metadata_list:
            if max_samples is not None and len(samples) >= max_samples:
                break

            # Check invalid list FIRST (in-memory, instant) before any filesystem access
            video_key = str(Path(item['relative_path']) / "video.mp4")
            if self.invalid_videos is not None and video_key in self.invalid_videos:
                continue

            video_path = self.root / item['relative_path'] / "video.mp4"

            rephrased = item.get('rephrased_instruction', '')
            if not rephrased:
                rephrased = item.get('instruction', '')

            start_frame = item.get('start', None)
            end_frame = item.get('end', None)

            if start_frame is not None and end_frame is not None:
                num_frames = end_frame - start_frame + 1
            else:
                video_key = str(Path(item['relative_path']) / "video.mp4")
                if video_key not in self.frame_counts:
                    continue
                num_frames = self.frame_counts[video_key]

            samples.append({
                'video_path': str(video_path),
                'instruction': rephrased,
                'primary_object': item.get('primary_object', ''),
                'all_objects': item.get('all_objects', ''),
                'dataset': 'hand_pretrain',
                'relative_path': item['relative_path'],
                'num_frames': num_frames,
                'start_frame': start_frame if start_frame is not None else 0,
                'end_frame': end_frame if end_frame is not None else num_frames - 1,
            })

        return samples

    def _load_real_robot_samples(self, max_samples: Optional[int] = None) -> List[Dict]:
        """Load real robot samples from Muse_Robot_Dataset structure.
        
        Each annotation in a video creates a separate sample with:
        - start/end frame indices from annotation
        - sensor token determined by instruction and annotation index
        - primary_object determined by instruction content
        - Only loads data_types specified in self.data_mix_sensors
        """
        samples = []
        
        # Map data_mix sensor names to data folder names
        sensor_to_folder = {
            'acoustic': 'acoustic_data',
            'radar': 'radar_data',
            'thermal': 'thermal_data',
        }
        
        # Filter data types based on data_mix_sensors
        data_types_to_load = []
        for sensor in self.data_mix_sensors:
            folder_name = sensor_to_folder.get(sensor)
            if folder_name:
                data_types_to_load.append(folder_name)
        
        if not data_types_to_load:
            logger.warning(f"No valid data types found in data_mix: {self.data_mix_sensors}")
            return []
        
        # Iterate through filtered data type folders
        for data_type in data_types_to_load:
            data_folder = self.root / data_type
            if not data_folder.exists():
                logger.warning(f"Data folder not found: {data_folder}")
                continue
            
            label_folder = self.root / data_type.replace('_data', '_labels')
            
            # Iterate through episode folders
            episode_folders = sorted([d for d in data_folder.iterdir() if d.is_dir()])
            for episode_folder in episode_folders:
                if max_samples is not None and len(samples) >= max_samples:
                    break
                
                episode_name = episode_folder.name
                
                # Check validity file
                validity_file = label_folder / f"{episode_name}_validity.json"
                if validity_file.exists():
                    try:
                        with open(validity_file, 'r') as f:
                            validity_data = json.load(f)
                        if validity_data.get('abolished', False) and validity_data.get('reason', None) == 'Already abolished in dataset label file':
                            continue
                    except Exception as e:
                        logger.warning(f"Could not read validity file {validity_file}: {e}")
                
                # Check label file for annotations
                label_file = label_folder / f"{episode_name}.json"
                if not label_file.exists():
                    logger.warning(f"Label file not found: {label_file}")
                    continue
                
                try:
                    with open(label_file, 'r') as f:
                        label_data = json.load(f)
                    
                    # Check if episode is abolished in label file
                    if label_data.get('problem', {}).get('abolished', False):
                        continue
                    
                    # Get annotations
                    annotations = label_data.get('annotations', [])
                    if not annotations:
                        logger.info(f"No annotations found for {episode_name}")
                        continue
                    
                    # Get total frame count from label file
                    total_frames = label_data.get('frame_num_total', None)
                    
                except Exception as e:
                    logger.warning(f"Could not read label file {label_file}: {e}")
                    continue
                
                # Check if video.mp4 exists
                video_path = episode_folder / "video.mp4"
                if not video_path.exists():
                    logger.warning(f"Video not found: {video_path}")
                    continue
                
                # # If frame count not in label file, probe video directly
                # if total_frames is None:
                #     import cv2
                #     cap = cv2.VideoCapture(str(video_path))
                #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                #     cap.release()
                
                # Create a sample for each annotation
                for ann_idx, annotation in enumerate(annotations):
                    if max_samples is not None and len(samples) >= max_samples:
                        break
                    
                    instruction = annotation.get('instruction', '')
                    start_frame = annotation.get('start', 0)
                    end_frame = annotation.get('end', total_frames - 1)
                    
                    # Calculate frame number for this annotation
                    num_frames = end_frame - start_frame + 1
                    
                    # Determine sensor token and primary_object based on data_type and annotation index
                    instruction_lower = instruction.lower()
                    if data_type == 'acoustic_data':
                        if 'towel' in instruction_lower:
                            primary_object = 'towels'
                        elif 'cloth' in instruction_lower:
                            primary_object = 'clothes'
                        else:
                            primary_object = 'towels'
                    elif data_type == 'thermal_data':
                        primary_object = 'bottles'
                    elif data_type == 'radar_data':
                        primary_object = 'boxes'
                    else:
                        primary_object = 'none'
                    if ann_idx == 0 and 'place' not in instruction_lower:
                        # First annotation and not a placement task: determine from data type
                        if data_type == 'acoustic_data':
                            sensor_token = 'audio'
                        elif data_type == 'thermal_data':
                            sensor_token = 'thermal'
                        elif data_type == 'radar_data':
                            sensor_token = 'radar'
                        else:
                            sensor_token = 'none'
                    else:
                        # Second+ annotation or placement task: no sensor
                        sensor_token = 'none'
                    
                    samples.append({
                        'video_path': str(video_path),
                        'instruction': instruction,
                        'dataset': 'real_robot',
                        'data_type': data_type,
                        'episode_name': episode_name,
                        'relative_path': f"{data_type}/{episode_name}",  # For VITRA-style action loading
                        'num_frames': num_frames,
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'sensor_token': sensor_token,
                        'primary_object': primary_object,
                        'annotation_index': ann_idx,
                    })
                
                if max_samples is not None and len(samples) >= max_samples:
                    break
            
            if max_samples is not None and len(samples) >= max_samples:
                break
        
        return samples

    def _load_real_robot_merged_samples(self, max_samples: Optional[int] = None) -> List[Dict]:
        """Load real robot samples from multiple Muse_Robot_Dataset directories.

        Iterates over self.root and self.extra_roots, scanning all *_data/ subdirectories.
        Uses INSTRUCTION_SENSOR_MAP to determine sensor_token and primary_object
        from annotation instruction text (instead of inferring from folder name).

        Filtering behavior controlled by self.use_validity_filter:
        - Always checks label.problem.abolished
        - If use_validity_filter=True: also checks _validity.json
        - If use_validity_filter=False: skips validity file

        If a cached metadata file (real_robot_merged_metadata_cache.json) exists in
        self.root, it is loaded directly to avoid thousands of filesystem I/O ops.
        The cache stores video_path with the original generation-time prefix; these
        are rebased to self.root / self.extra_roots at load time.
        """
        # --- Try loading from cache first ---
        cache_path = self.root / "real_robot_merged_metadata_cache.json"
        if cache_path.exists():
            logger.info(f"[real_robot_merged] Loading cached metadata from {cache_path}")
            import time as _time
            _t0 = _time.time()
            with open(cache_path, 'r') as f:
                samples = json.load(f)

            # Rebase video_path: the cache was generated with a specific root prefix
            # (e.g. ./mount<CHECKPOINT_PATH> but at runtime the
            # root may differ (e.g. <DATA_ROOT>
            # Each sample has root_index indicating which root it came from (0=primary,
            # 1=first extra_root, etc.). We reconstruct video_path accordingly.
            all_roots = [self.root] + self.extra_roots
            for sample in samples:
                rel = sample['relative_path']
                ri = sample.get('root_index', 0)
                root = all_roots[ri] if ri < len(all_roots) else all_roots[0]
                sample['video_path'] = str(root / rel / "video.mp4")

            if max_samples is not None:
                samples = samples[:max_samples]
            _elapsed = _time.time() - _t0
            logger.info(f"[real_robot_merged] Loaded {len(samples)} cached samples in {_elapsed:.1f}s")
            return samples

        logger.info("[real_robot_merged] No metadata cache found, scanning filesystem (this may be slow on mounted storage)...")

        samples = []
        all_roots = [self.root] + self.extra_roots

        for root_idx, root_path in enumerate(all_roots):
            root_path = Path(root_path)
            if not root_path.exists():
                logger.warning(f"Dataset root not found: {root_path}")
                continue

            # Find all *_data/ subdirectories
            logger.info(f"[real_robot_merged] Scanning root {root_idx+1}/{len(all_roots)}: {root_path}")
            data_folders = sorted([
                d for d in root_path.iterdir()
                if d.is_dir() and d.name.endswith('_data')
            ])

            if not data_folders:
                logger.warning(f"No *_data/ folders found in {root_path}")
                continue

            for data_folder in data_folders:
                label_folder = root_path / data_folder.name.replace('_data', '_labels')

                logger.info(f"[real_robot_merged] Listing episodes in {data_folder.name}...")
                episode_folders = sorted([d for d in data_folder.iterdir() if d.is_dir()])
                logger.info(f"[real_robot_merged] Found {len(episode_folders)} episodes in {data_folder.name}")
                for ep_i, episode_folder in enumerate(episode_folders):
                    if max_samples is not None and len(samples) >= max_samples:
                        break

                    if (ep_i + 1) % 200 == 0:
                        logger.info(f"[real_robot_merged] {data_folder.name}: {ep_i+1}/{len(episode_folders)} episodes, {len(samples)} samples so far")

                    episode_name = episode_folder.name

                    # --- Filtering ---
                    # Load label file
                    label_file = label_folder / f"{episode_name}.json"
                    if not label_file.exists():
                        logger.warning(f"Label file not found: {label_file}")
                        continue

                    try:
                        with open(label_file, 'r') as f:
                            label_data = json.load(f)
                    except Exception as e:
                        logger.warning(f"Could not read label file {label_file}: {e}")
                        continue

                    # Always check label abolished
                    if label_data.get('problem', {}).get('abolished', False):
                        continue

                    # Optionally check validity file
                    if self.use_validity_filter:
                        validity_file = label_folder / f"{episode_name}_validity.json"
                        if validity_file.exists():
                            try:
                                with open(validity_file, 'r') as f:
                                    validity_data = json.load(f)
                                if validity_data.get('abolished', False):
                                    continue
                            except Exception as e:
                                logger.warning(f"Could not read validity file {validity_file}: {e}")

                    # Get annotations
                    annotations = label_data.get('annotations', [])
                    if not annotations:
                        logger.info(f"No annotations found for {episode_name}")
                        continue

                    total_frames = label_data.get('frame_num_total', None)

                    # Check video exists
                    video_path = episode_folder / "video.mp4"
                    if not video_path.exists():
                        logger.warning(f"Video not found: {video_path}")
                        continue

                    # Create a sample for each annotation
                    for ann_idx, annotation in enumerate(annotations):
                        if max_samples is not None and len(samples) >= max_samples:
                            break

                        instruction = annotation.get('instruction', '')
                        start_frame = annotation.get('start', 0)
                        end_frame = annotation.get('end', total_frames - 1)
                        num_frames = end_frame - start_frame + 1

                        # Look up sensor_token and primary_object from mapping
                        if instruction in INSTRUCTION_SENSOR_MAP:
                            sensor_token, primary_object = INSTRUCTION_SENSOR_MAP[instruction]
                        else:
                            logger.warning(
                                f"Instruction not in INSTRUCTION_SENSOR_MAP: '{instruction}' "
                                f"(episode={episode_name}, ann_idx={ann_idx}). Using fallback."
                            )
                            sensor_token, primary_object = ("none", "unknown")

                        samples.append({
                            'video_path': str(video_path),
                            'instruction': instruction,
                            'dataset': 'real_robot',  # reuse existing downstream logic
                            'data_type': data_folder.name,
                            'episode_name': episode_name,
                            'relative_path': f"{data_folder.name}/{episode_name}",
                            'num_frames': num_frames,
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'sensor_token': sensor_token,
                            'primary_object': primary_object,
                            'annotation_index': ann_idx,
                        })

                    if max_samples is not None and len(samples) >= max_samples:
                        break

                if max_samples is not None and len(samples) >= max_samples:
                    break

            if max_samples is not None and len(samples) >= max_samples:
                break

        logger.info(f"Loaded {len(samples)} merged real_robot samples from {len(all_roots)} roots")
        return samples

    def _load_molmoact_samples(self, max_samples: Optional[int] = None) -> List[Dict]:
        """Load MolmoAct dataset samples."""
        # Determine split (household or tabletop)
        split = self.dataset_name.split('_')[1]  # household or tabletop
        
        samples = []
        for split_dir in ['train']:
            # Process videos and use cached frame counts
            meta_dir = self.root / split_dir / "meta"
            
            # Load extra_sensor.json for task metadata
            metadata_file = meta_dir / "extra_sensor.json"
            if not metadata_file.exists():
                continue
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            
            # Load tasks.jsonl to get task_index -> task_description mapping
            tasks_file = meta_dir / "tasks.jsonl"
            if not tasks_file.exists():
                logger.warning(f"tasks.jsonl not found in {meta_dir}")
                continue
            
            tasks = {}
            with open(tasks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    task_data = json.loads(line.strip())
                    tasks[task_data['task_index']] = task_data['task']
            
            # Create reverse mapping
            task_desc_to_idx = {desc: idx for idx, desc in tasks.items()}
            
            # Load episodes.jsonl to map episodes to tasks
            episodes_file = meta_dir / "episodes.jsonl"
            if not episodes_file.exists():
                logger.warning(f"episodes.jsonl not found in {meta_dir}")
                continue
            
            # Build mapping of task_index -> list of episode indices
            task_to_episodes = {}
            with open(episodes_file, 'r', encoding='utf-8') as f:
                for line in f:
                    episode_data = json.loads(line.strip())
                    episode_idx = episode_data['episode_index']
                    task_list = episode_data['tasks']
                    
                    if task_list:
                        task_desc = task_list[0]
                        task_idx = task_desc_to_idx.get(task_desc)
                        
                        if task_idx is not None:
                            if task_idx not in task_to_episodes:
                                task_to_episodes[task_idx] = []
                            task_to_episodes[task_idx].append(episode_idx)
            
            # Create samples for each episode
            for task_key, task_meta in metadata_dict.items():
                if max_samples is not None and len(samples) >= max_samples:
                    break
                
                task_index = task_meta.get('task_index')
                rephrased = task_meta.get('rephrased_instruction', '')
                
                episode_indices = task_to_episodes.get(task_index, [])
                
                for episode_idx in episode_indices:
                    if max_samples is not None and len(samples) >= max_samples:
                        break
                    
                    # Build video path
                    chunk_num = episode_idx // 1000
                    chunk_name = f"chunk-{chunk_num:03d}"
                    episode_name = f"episode_{episode_idx:06d}"
                    
                    video_path = self.root / split_dir / "videos" / chunk_name / "observations.rgb" / f"{episode_name}.mp4"
                    
                    if not video_path.exists():
                        continue
                    
                    # Skip if video is in invalid videos list (if cache exists)
                    video_key = str(video_path.relative_to(self.root))
                    if self.invalid_videos is not None and video_key in self.invalid_videos:
                        continue
                    if video_key not in self.frame_counts:
                        logger.warning(f"Frame count not found for {video_key}")
                        continue
                    
                    num_frames = self.frame_counts[video_key]
                    
                    samples.append({
                        'video_path': str(video_path),
                        'instruction': rephrased,
                        'primary_object': task_meta.get('primary_object', ''),
                        'all_objects': task_meta.get('all_objects', ''),
                        'dataset': f'molmoact_{split}',
                        'episode_idx': episode_idx,
                        'chunk_name': chunk_name,
                        'episode_name': episode_name,
                        'num_frames': num_frames,
                    })
        
        return samples
    
    def _load_agibot_samples(self, max_samples: Optional[int] = None) -> List[Dict]:
        """Load AgiBotWorld dataset samples (JSONL format)."""
        metadata_file = self.root / "task_info" / "extra_sensor.jsonl"
        if not metadata_file.exists():
            logger.warning(f"Metadata file not found: {metadata_file}")
            return []
        
        samples = []
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                # Early termination if max_samples reached
                if max_samples is not None and len(samples) >= max_samples:
                    break
                    
                item = json.loads(line)
                
                task_id = item.get('task_id', '')
                episode_id = item.get('episode_id', '')
                
                video_path = self.root / "observations_extract" / task_id / episode_id / "videos" / "head_color_h264.mp4"
                if not video_path.exists():
                    continue
                
                # Skip if video is in invalid videos list (if cache exists)
                video_key = str(video_path.relative_to(self.root))
                if self.invalid_videos is not None and video_key in self.invalid_videos:
                    continue
                if video_key not in self.frame_counts:
                    logger.warning(f"Frame count not found for {video_key}")
                    continue
                
                num_frames = self.frame_counts[video_key]
                
                samples.append({
                    'video_path': str(video_path),
                    'instruction': item.get('rephrased_instruction', item.get('original_instruction', '')),
                    'primary_object': item.get('primary_object', ''),
                    'all_objects': item.get('all_objects', ''),
                    'dataset': 'agibot',
                    'task_id': task_id,
                    'episode_id': episode_id,
                    'num_frames': num_frames,
                })
        
        return samples
    
    def _sample_sensor_token(self, sample: Optional[Dict] = None) -> str:
        """Sample a sensor token based on configured probabilities.
        
        For real_robot dataset, the sensor token is already determined in the sample.
        For other datasets, sample randomly based on probabilities.
        """
        # If RGB_only mode, never use a sensor
        if self.RGB_only:
            return 'none'

        # For real_robot, sensor token is pre-determined
        if sample and sample.get('dataset') == 'real_robot':
            return sample.get('sensor_token', 'none')
        
        # For vitra with random_sample_sensor=False, always return 'none'
        if not self.random_sample_sensor and sample and sample.get('dataset') == 'vitra':
            return 'none'
        
        # For other datasets, sample randomly
        tokens = list(self.sensor_token_prob.keys())
        probs = list(self.sensor_token_prob.values())
        return random.choices(tokens, weights=probs, k=1)[0]
    
    def _replace_placeholders(self, text: str, sensor_type: str, semantic_label: str) -> str:
        """Replace placeholder text with sampled words from probability distributions.

        Args:
            text: Text containing placeholder to replace
            sensor_type: Sensor type (thermal/radar/audio/none)
            semantic_label: Semantic category (e.g., 'cold', 'hot', 'hidden', etc.)

        Returns:
            Text with placeholder replaced by sampled word from the appropriate category
        """
        if self.placeholder_text not in text:
            return text

        # For 'none' sensor, remove placeholder
        if sensor_type == 'none':
            return text.replace(self.placeholder_text + ' ', '').replace(self.placeholder_text, '').strip()

        if not self.word_probs:
            return text

        # Sample word from the specific sensor type and semantic label category
        if sensor_type in self.word_probs and semantic_label in self.word_probs[sensor_type]:
            word_dict = self.word_probs[sensor_type][semantic_label]
            if word_dict:
                words = list(word_dict.keys())
                probs = list(word_dict.values())
                # Probabilities should already be normalized, but ensure it
                total = sum(probs)
                if total > 0:
                    probs = [p/total for p in probs]
                    replacement = random.choices(words, weights=probs, k=1)[0]

                    # Special handling for radar "hidden" category words that describe
                    # occlusion (hidden, covered, occluded) - these need phrase restructuring
                    # e.g., "Pick the <placeholder> box" -> "Pick the box with a hidden item"
                    # instead of "Pick the hidden box" which is grammatically awkward
                    occlusion_words = {'hidden', 'covered', 'occluded'}
                    if replacement in occlusion_words:
                        import re
                        # Match pattern: "<placeholder> <noun>" and restructure to "<noun> with a <word> item"
                        pattern = rf'{re.escape(self.placeholder_text)}\s+(\w+)'
                        match = re.search(pattern, text)
                        if match:
                            noun = match.group(1)
                            restructured = f'{noun} with a {replacement} item'
                            return re.sub(pattern, restructured, text)

                    return text.replace(self.placeholder_text, replacement)

        # Fallback: remove placeholder if no word dict found
        return text.replace(self.placeholder_text + ' ', '').replace(self.placeholder_text, '').strip()
    
    def _generalize_object_name(self, instruction: str) -> str:
        """Replace specific object nouns with generic 'item' in instruction.
        
        Forces model to rely on sensor information rather than object identity.
        E.g., "Pick up warm mango." → "Pick up warm item."
        
        The adjective list is the exhaustive union of all words from the
        thermal/acoustic/radar word_probs JSON files.  Keep in sync when
        adding new sensor words.
        """
        import re
        # Handle two-word phrase "room temperature" first
        result = re.sub(
            r'(room temperature\s+)'
            r'(\S+(?:\s+\S+)*?)'
            r'(\s*(?:into|in|onto|on|from|to|covering)\b|\.|\s*$)',
            r'\1item\3',
            instruction,
            flags=re.IGNORECASE,
        )
        # Then handle all single-word sensor adjectives (exhaustive from word dicts)
        result = re.sub(
            r'((?:freezing|icy|frosty|subzero|frozen|frostbitten|cold|chilly|chill|brisk|nippy|'
            r'cool|mild|tepid|lukewarm|ambient|'
            r'warm|toasty|hot|heated|blazing|scalding|roasting|searing|overheated|'
            r'filled|occupied|stuffed|loaded|hidden|occluded|covered|empty|vacant|hollow|void|'
            r'loud|noisy|ringing|silent|quiet|hushed)\s+)'
            r'(\S+(?:\s+\S+)*?)'
            r'(\s*(?:into|in|onto|on|from|to|covering)\b|\.|\s*$)',
            r'\1item\3',
            result,
            flags=re.IGNORECASE,
        )
        return result
    
    def _get_clone_aug_suffix(self, episode_dir: Path) -> Optional[str]:
        """Determine which clone augmentation suffix is available for this episode.
        
        Checks for new format (_aug0/_aug1) and old format (_aug).
        Also verifies that sensor aug files actually exist in subdirs
        (not just the root video_aug*.mp4), to avoid loading missing files.
        Returns a randomly chosen available suffix, or None if none exist.
        """
        available = []
        # Check new format: _aug0, _aug1
        for i in range(2):
            suffix = f"_aug{i}"
            if (episode_dir / f"video{suffix}.mp4").exists():
                # Spot-check: verify at least one sensor aug file exists per mode
                ok = True
                for subdir in ('thermal', 'radar', 'audio'):
                    sub = episode_dir / subdir
                    if not sub.exists():
                        ok = False
                        break
                    # Check any file with this suffix exists in subdir
                    found = any(f.name.endswith(f'{suffix}.mp4') for f in sub.iterdir() if f.is_file())
                    if not found:
                        ok = False
                        break
                if ok:
                    available.append(suffix)
        # Check old format: _aug
        if (episode_dir / "video_aug.mp4").exists():
            ok = True
            for subdir in ('thermal', 'radar', 'audio'):
                sub = episode_dir / subdir
                if not sub.exists():
                    ok = False
                    break
                found = any(f.name.endswith('_aug.mp4') for f in sub.iterdir() if f.is_file())
                if not found:
                    ok = False
                    break
            if ok:
                available.append("_aug")
        
        if not available:
            return None
        return random.choice(available)
    
    def _get_sensor_video_path(self, sample: Dict, sensor_type: str, semantic_label: str) -> Optional[Path]:
        """
        Get path to sensor overlay video based on sensor type and semantic label.
        Can return a single path or a dict of paths for all sensors when sensor_input_all=True.
        
        Dataset-specific path patterns:
        - VITRA: {video_dir}/{sensor_type}/{video_name}_{semantic_label}_{background_mode}.mp4
        - Agibot: {episode_dir}/{sensor_type}/{video_name}_{semantic_label}_{background_mode}.mp4
        
        Returns:
            If sensor_input_all=True: Dict[str, Path] with keys 'thermal', 'radar', 'acoustic'
            If sensor_input_all=False: Optional[Path] for single sensor
        """
        if sensor_type == "none" and not self.sensor_input_all:
            return None
        
        # If sensor_input_all, load all three sensors regardless of sensor_type
        if self.sensor_input_all:
            sensor_paths = {}
            for sensor in ['thermal', 'radar', 'acoustic']:
                # Use appropriate semantic label for each sensor
                if sensor in self.word_probs and len(self.word_probs[sensor]) > 0:
                    categories = list(self.word_probs[sensor].keys())
                    sensor_semantic_label = random.choice(categories)
                else:
                    # Fallback labels
                    fallback_labels = {
                        'thermal': ['cold', 'room_temperature', 'hot'],
                        'radar': ['hidden', 'absense'],
                        'acoustic': ['ringing', 'silence'],
                    }
                    sensor_semantic_label = random.choice(fallback_labels.get(sensor, ['default']))
                
                sensor_paths[sensor] = self._get_single_sensor_video_path(sample, sensor, sensor_semantic_label)
            return sensor_paths
        else:
            # Single sensor mode
            return self._get_single_sensor_video_path(sample, sensor_type, semantic_label)
    
    def _get_single_sensor_video_path(self, sample: Dict, sensor_type: str, semantic_label: str) -> Optional[Path]:
        """
        Get path to a single sensor overlay video.
        
        Args:
            sample: Sample metadata
            sensor_type: Sensor type (thermal/radar/acoustic/audio)
            semantic_label: Semantic category for the sensor
            
        Returns:
            Path to sensor video file
        """
        video_path = Path(sample['video_path'])
        video_name = video_path.stem
        dataset_type = sample['dataset']
        background_mode = self.background_modes
        
        # Construct dataset-specific sensor video path
        if dataset_type in ('vitra', 'hand_pretrain'):
            # VITRA / hand_pretrain: video_dir / sensor_type / filename
            sensor_base_dir = video_path.parent
            sensor_video_path = sensor_base_dir / sensor_type / f"{video_name}_{semantic_label}_{background_mode}.mp4"
            
        elif dataset_type == 'real_robot':
            # Real robot: episode_dir / sensor_type / images_masked / video.mp4 (or video_dark.mp4, video_grey.mp4)
            episode_dir = video_path.parent
            
            # Map sensor token to folder name
            sensor_folder_map = {
                'audio': 'acoustic',
                'radar': 'radar',
                'thermal': 'thermal',
            }
            sensor_folder = sensor_folder_map.get(sensor_type, sensor_type)
            
            # Determine video filename based on background mode and sensor_masked flag
            if self.sensor_masked:
                # Use masked version
                if background_mode == 'dark':
                    video_filename = 'video_dark.mp4'
                elif background_mode == 'grey':
                    video_filename = 'video_grey.mp4'
                else:  # color
                    video_filename = 'video.mp4'
                # For radar with radar_fix_start_frame, use the first-frame-fixed masked folder
                if sensor_folder == 'radar' and self.radar_fix_start_frame:
                    sensor_video_path = episode_dir / sensor_folder / 'images_masked_first_frame' / video_filename
                else:
                    sensor_video_path = episode_dir / sensor_folder / 'images_masked' / video_filename
            else:
                # Use raw sensor images (combined for thermal)
                sensor_video_path = episode_dir / sensor_folder / 'images' / 'video.mp4'
            
        elif dataset_type == 'agibot':
            # Agibot: episode_dir / sensor_type / filename
            # video_path: observations_extract/task_id/episode_id/videos/head_color_h264.mp4
            # sensor_path: observations_extract/task_id/episode_id/sensor_type/head_color_{semantic_label}_{background_mode}.mp4
            episode_dir = video_path.parent.parent  # Go up from videos/ to episode_dir
            clean_video_name = video_name.replace('_h264', '')
            sensor_video_path = episode_dir / sensor_type / f"{clean_video_name}_{semantic_label}_{background_mode}.mp4"
            
        elif 'molmoact' in dataset_type:
            # MolmoAct: chunk_dir / observations.sensor / sensor_type / filename
            # video_path: train/videos/chunk-XXX/observations.rgb/episode_YYYYYY.mp4
            # sensor_path: train/videos/chunk-XXX/observations.sensor/sensor_type/episode_YYYYYY_{semantic_label}_{background_mode}.mp4
            chunk_dir = video_path.parent.parent  # Go up from observations.rgb/ to chunk-XXX/
            sensor_base_dir = chunk_dir / "observations.sensor"
            sensor_video_path = sensor_base_dir / sensor_type / f"{video_name}_{semantic_label}_{background_mode}.mp4"
            
        else:
            return None
        
        return sensor_video_path
        
        # if not sensor_video_path.exists():
        #     logging.warning(f"Sensor video not found: {sensor_video_path}")
        #     return None
        # else:
        #     return sensor_video_path

    def _get_seg_token_str(self, video_path: str, frame_idx: int, dataset_name: str) -> str:
        """Return PaliGemma-style loc+seg token string for a given video frame.

        frame_idx is an absolute video frame index (same as used when reading the RGB video),
        which maps 1-to-1 to the token file since n == total video frames.
        """
        vp = Path(video_path)
        if dataset_name in ('vitra', 'hand_pretrain'):
            token_path = vp.parent / "video_seg_tokens.npz"
        elif dataset_name in ('real_robot', 'real_robot_old'):
            token_path = vp.parent / "mask" / "video_tokens.npz"
        elif dataset_name == 'agibot':
            token_path = vp.parent.parent / "head_mask_tokens.npz"
        else:
            return ''

        if not token_path.exists():
            raise FileNotFoundError(f"Seg token file not found: {token_path}")

        data = np.load(str(token_path))
        n = len(data['loc_tokens'])
        if n == 0:
            raise ValueError(f"Seg token file is empty: {token_path}")
        if int(frame_idx) > n - 1:
            raise IndexError(f"frame_idx {frame_idx} out of range for token file with {n} frames: {token_path}")

        fi = int(frame_idx)
        loc = data['loc_tokens'][fi]   # [4] int32
        seg = data['seg_tokens'][fi]   # [16] int32
        return ''.join(f'<loc{int(v):04d}>' for v in loc) + ''.join(f'<seg{int(v):03d}>' for v in seg)

    def _load_rgb_frame(self, video_path: str, frame_idx: int) -> Optional[np.ndarray]:
        """Load a single RGB frame from video with timeout protection."""
        frame = load_frame_with_timeout(video_path, frame_idx)

        if frame is None:
            return None

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    def _load_sensor_frame(self, sensor_video_path, frame_idx: int, sample: Optional[Dict] = None):
        """Load sensor overlay frame(s).
        
        Args:
            sensor_video_path: Single Path or Dict[str, Path] for multiple sensors
            frame_idx: RGB frame index (0-based)
            sample: Sample metadata
            
        Returns:
            If sensor_video_path is dict: Dict[str, Optional[np.ndarray]] with sensor frames
            If sensor_video_path is Path: Optional[np.ndarray] with single sensor frame
            If sensor_video_path is None: None
        """
        if sensor_video_path is None:
            # Return None for "none" sensor
            return None
        
        # Handle dict of paths (sensor_input_all mode)
        if isinstance(sensor_video_path, dict):
            sensor_frames = {}
            for sensor_name, path in sensor_video_path.items():
                sensor_frames[sensor_name] = self._load_single_sensor_frame(path, frame_idx, sample)
            return sensor_frames
        else:
            # Single sensor path
            return self._load_single_sensor_frame(sensor_video_path, frame_idx, sample)
    
    def _load_single_sensor_frame(self, sensor_video_path: Optional[Path], frame_idx: int, sample: Optional[Dict] = None) -> Optional[np.ndarray]:
        """Load a single sensor overlay frame from one video file with timeout protection.

        Args:
            sensor_video_path: Path to sensor video file
            frame_idx: RGB frame index (0-based)
            sample: Sample metadata (unused, kept for interface compatibility)
        """
        if sensor_video_path is None:
            # Return None for "none" sensor
            return None

        video_path_str = str(sensor_video_path)

        # Get frame count with timeout protection
        frame_count = get_video_frame_count_with_timeout(video_path_str)
        if frame_count == 0:
            logger.warning(f"Video has 0 frames or failed to open: {sensor_video_path}")
            return None

        sensor_frame_idx = frame_idx
        if sensor_frame_idx >= frame_count:
            logger.warning(f"Sensor frame index {sensor_frame_idx} out of range for video with {frame_count} frames: {sensor_video_path}")
            return None

        # Load frame with timeout protection
        frame = load_frame_with_timeout(video_path_str, sensor_frame_idx)

        if frame is None:
            logger.warning(f"Failed to read sensor frame {sensor_frame_idx} (RGB frame {frame_idx}) from {sensor_video_path}")
            return None

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    def _load_actions_and_state(self, sample: Dict, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load action trajectory and current state based on dataset type.
        
        Returns:
            action_sequence: [chunk_size, action_dim] normalized actions
            action_mask: [chunk_size, action_dim] action validity mask
            current_state: [state_dim] normalized current state
            current_state_mask: [state_dim] state validity mask
        """
        dataset_type = sample['dataset']
        
        if dataset_type in ('vitra', 'real_robot', 'real_robot_old', 'hand_pretrain'):
            # Real robot and hand_pretrain use same structure as VITRA
            return self._load_vitra_actions(sample, frame_idx)
        elif 'molmoact' in dataset_type:
            return self._load_molmoact_actions(sample, frame_idx)
        elif dataset_type == 'agibot':
            return self._load_agibot_actions(sample, frame_idx)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
 

    def _load_general_actions_state(
        self,
        hand_state_data: list,
        ee_action_data: list,
        pose_data_list,
        arm_state_data,
        actual_frame_idx: int,
        episode_start: int,
        episode_end: Optional[int],
        ee_dim: int = 12,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
        """Build action/state arrays and extract action sequence.

        Handles both pose mode (use_end_pose=True) and joint mode (use_end_pose=False),
        then performs temporal sampling with adaptive jump frame.
        """
        # Mode 1: Position-based (6D end pose)
        if self.use_end_pose:
            # Combine action data: 6D pose + 12D hand
            action_data = []
            state_data = []  # For current state

            for i in range(len(pose_data_list)):
                # Current pose for state
                pose_dict = pose_data_list[i]
                pose_6d_current = [
                    pose_dict['x'],
                    pose_dict['y'],
                    pose_dict['z'],
                    pose_dict['rx'],
                    pose_dict['ry'],
                    pose_dict['rz']
                ]

                # Arm action is next pose (t+1) or delta pose depending on use_delta_pose
                if i + 1 < len(pose_data_list):
                    pose_dict_next = pose_data_list[i + 1]
                    pose_6d_next = np.array([
                        pose_dict_next['x'],
                        pose_dict_next['y'],
                        pose_dict_next['z'],
                        pose_dict_next['rx'],
                        pose_dict_next['ry'],
                        pose_dict_next['rz']
                    ])

                    if self.use_delta_pose:
                        # FIX: Use proper rotation delta calculation with rotation matrices
                        # Translation delta: simple subtraction is correct
                        t_delta = (pose_6d_next[:3] - np.array(pose_6d_current[:3])).tolist()

                        # Rotation delta: R_rel = R_{t+1} * R_t^{-1} (NOT simple Euler subtraction!)
                        # Euler angle subtraction is mathematically incorrect because Euler angles
                        # are non-linear and don't compose additively.
                        R_curr = R.from_euler('xyz', pose_6d_current[3:6], degrees=False)
                        R_next = R.from_euler('xyz', pose_6d_next[3:6], degrees=False)
                        R_rel = R_next * R_curr.inv()  # Proper relative rotation
                        rot_delta = R_rel.as_euler('xyz', degrees=False).tolist()
                        pose_6d_action = t_delta + rot_delta
                    else:
                        # Use absolute next pose
                        pose_6d_action = pose_6d_next.tolist()
                else:
                    # Last frame: use zeros for delta or current pose for absolute
                    if self.use_delta_pose:
                        pose_6d_action = [0.0] * 6
                    else:
                        pose_6d_action = pose_6d_current

                # EE action is from ee_command
                ee_action = ee_action_data[i] if i < len(ee_action_data) else [0.0] * ee_dim

                action_data.append(pose_6d_action + ee_action)  # 6 + ee_dim dimensions
                state_data.append(pose_6d_current + (hand_state_data[i] if i < len(hand_state_data) else [0.0] * ee_dim))  # 6 + ee_dim dimensions

            arm_dim = 6
            total_dim = arm_dim + ee_dim

        # Mode 2: Joint-based (7D joint angles)
        else:
            # Load state data (used for both current state and arm action target)
            # For VITRA, arm action is represented by state_(t+1) from sensor readings

            # Combine action data: arm action is state_(t+1) or delta, ee action is from ee_command
            action_data = []
            for i in range(len(arm_state_data)):
                # Arm action is next arm state (t+1) or delta depending on use_delta_pose
                if i + 1 < len(arm_state_data):
                    if self.use_delta_pose:
                        # Use delta: state_(t+1) - state_t
                        arm_action = (np.array(arm_state_data[i + 1]) - np.array(arm_state_data[i])).tolist()
                    else:
                        # Use absolute next state
                        arm_action = arm_state_data[i + 1]
                else:
                    # Last frame: use zeros for delta or current state for absolute
                    if self.use_delta_pose:
                        arm_action = [0.0] * 7
                    else:
                        arm_action = arm_state_data[i]

                # EE action is from ee_command
                ee_action = ee_action_data[i] if i < len(ee_action_data) else [0.0] * ee_dim
                action_data.append(arm_action + ee_action)  # 7 + ee_dim dimensions

            # Combine state data: arm + hand
            state_data = [arm + hand for arm, hand in zip(arm_state_data, hand_state_data)]  # 7 + ee_dim dimensions

            arm_dim = 7
            total_dim = 7 + ee_dim

        if episode_end is None:
            episode_end = len(action_data) - 1

        # Extract action sequence with temporal downsampling and adaptive jump frame
        # Similar to dataset_reference.py, handle boundaries: before start, valid range, after end
        target_samples = self.chunk_size
        action_list = []
        action_mask_list = []
        current_jump_frame = self.jump_frame
        switched_to_adaptive = False

        # Sample with fixed range similar to reference implementation
        for sample_idx in range(target_samples):
            i = actual_frame_idx + sample_idx * current_jump_frame

            # Check max_l1 condition before processing current frame
            max_l1 = 0.0
            if len(action_list) > 0 and not switched_to_adaptive and actual_frame_idx <= i < episode_end + 1:
                # Calculate max L1 difference on hand commands
                hand_start_idx = arm_dim
                hand_end_idx = hand_start_idx + ee_dim

                if i < len(action_data):
                    prev_action = action_list[-1]
                    curr_action = np.array(action_data[i], dtype=np.float32)
                    l1_diff = np.abs(curr_action[hand_start_idx:hand_end_idx] - prev_action[hand_start_idx:hand_end_idx])
                    max_l1 = np.max(l1_diff)

            # Check if we should switch to adaptive jump frame
            if max_l1 > self.threshold and not switched_to_adaptive:
                # Switch to adaptive_jump_frame for remaining samples
                current_jump_frame = self.adaptive_jump_frame
                switched_to_adaptive = True
                # Recalculate current frame index with new jump_frame
                i = actual_frame_idx + sample_idx * current_jump_frame

            # Handle three cases: before start, valid range, after end
            if i < episode_start:
                # Before episode start: use neutral action with mask=False
                action = np.zeros(total_dim, dtype=np.float32)
                action_list.append(action)
                action_mask_list.append(False)

            elif i >= episode_end + 1:
                # After episode end: repeat last valid action or use neutral
                if len(action_list) > 0:
                    last_valid_idx = len(action_list) - 1
                    last_action = action_list[last_valid_idx]
                    last_mask = action_mask_list[last_valid_idx] if last_valid_idx < len(action_mask_list) else False

                    if self.use_delta_pose:
                        if self.relative_to_t0:
                            # Cumulative from t0: robot stays at final position
                            # Keep the full last cumulative delta (arm + hand)
                            action = last_action.copy()
                        else:
                            # Frame-to-frame delta: no movement = zero arm delta, keep hand command
                            action = np.zeros(total_dim, dtype=np.float32)
                            action[arm_dim:] = last_action[arm_dim:]
                    else:
                        # For absolute pose: repeat last action
                        action = last_action.copy()

                    action_list.append(action)
                    # Mask is True only if within 8*current_jump_frame steps after end AND last frame was valid
                    mask = ((i - episode_end - 1) < 8 * current_jump_frame) and last_mask
                    action_mask_list.append(mask)
                else:
                    # No previous actions, use neutral
                    action = np.zeros(total_dim, dtype=np.float32)
                    action_list.append(action)
                    action_mask_list.append(False)

            else:
                # Within valid range: load action normally
                if i < len(action_data):
                    if self.use_delta_pose:
                        target_frame = min(i + current_jump_frame, len(state_data) - 1)

                        if self.relative_to_t0:
                            # Cumulative delta from observation frame (t0) to target frame
                            # Matches reference code's relative_to_t0=True approach
                            ref_pose = np.array(state_data[actual_frame_idx][:arm_dim], dtype=np.float32)
                        else:
                            # Frame-to-frame delta: reference is current frame i
                            # Matches reference code's relative_to_t0=False approach
                            ref_pose = np.array(state_data[i][:arm_dim], dtype=np.float32)

                        target_pose = np.array(state_data[target_frame][:arm_dim], dtype=np.float32)

                        # Translation delta
                        t_delta = target_pose[:3] - ref_pose[:3]

                        # Rotation: R_target * R_ref^{-1}
                        R_ref = R.from_euler('xyz', ref_pose[3:6], degrees=False)
                        R_target = R.from_euler('xyz', target_pose[3:6], degrees=False)
                        R_rel = R_target * R_ref.inv()
                        rot_delta = R_rel.as_euler('xyz', degrees=False).astype(np.float32)

                        arm_delta = np.concatenate([t_delta, rot_delta]).astype(np.float32)

                        # Hand command at target_frame - 1 (matching reference code)
                        hand_idx = max(target_frame - 1, 0)
                        ee_action = np.array(ee_action_data[hand_idx], dtype=np.float32) if hand_idx < len(ee_action_data) else np.zeros(ee_dim, dtype=np.float32)

                        action = np.concatenate([arm_delta, ee_action]).astype(np.float32)
                        action_list.append(action)
                    else:
                        action_list.append(np.array(action_data[i], dtype=np.float32))

                    action_mask_list.append(True)
                else:
                    # Beyond data length: use neutral
                    action = np.zeros(total_dim, dtype=np.float32)
                    action_list.append(action)
                    action_mask_list.append(False)

        # Convert to numpy arrays
        actions = np.array(action_list, dtype=np.float32)  # [chunk_size, total_dim]
        action_mask = np.array([[mask] * total_dim for mask in action_mask_list], dtype=bool)  # [chunk_size, total_dim]

        # Load current state
        current_state = np.array(state_data[actual_frame_idx], dtype=np.float32) if actual_frame_idx < len(state_data) else np.zeros(total_dim, dtype=np.float32)

        # Create state mask
        current_state_mask = np.ones(total_dim, dtype=bool)

        # if DEBUG_FLAG_EULER:
        #     print("[DEBUG] Fixed Euler angle handling for delta pose mode.")

        return actions, action_mask, current_state, current_state_mask, total_dim, arm_dim


   
    def _read_vitra_action_state(self, recording_dir: Path) -> Tuple:
        """Load raw VITRA/real_robot file data.

        Returns:
            (hand_state_data, ee_action_data, pose_data_list, arm_state_data)
            pose_data_list is None when use_end_pose=False
            arm_state_data is None when use_end_pose=True
        """
        # Auto-detect which hand side has data (right is default for VITRA,
        # but hand_pretrain may have left-only episodes)
        right_hs = recording_dir / "hand_status" / "right_hand_position.txt"
        left_hs = recording_dir / "hand_status" / "left_hand_position.txt"
        if right_hs.exists():
            side = "right"
        elif left_hs.exists():
            side = "left"
        else:
            raise FileNotFoundError(f"No hand_status file in {recording_dir}")

        hand_state_file = recording_dir / "hand_status" / f"{side}_hand_position.txt"
        hand_state_data = []
        with open(hand_state_file, 'r') as f:
            for line in f:
                values = [float(x) for x in line.strip().split()]
                hand_state_data.append(values)

        ee_action_file = recording_dir / "ee_command" / f"{side}_ee.txt"
        ee_action_data = []
        with open(ee_action_file, 'r') as f:
            for line in f:
                values = [float(x) for x in line.strip().split()]
                ee_action_data.append(values)

        if self.use_end_pose:
            pose_file = recording_dir / "6d_pose" / f"{side}_arm.npy"
            if not pose_file.exists():
                raise FileNotFoundError(f"6D pose file not found: {pose_file}")
            pose_data_list = np.load(pose_file, allow_pickle=True)
            arm_state_data = None
        else:
            arm_state_file = recording_dir / "arm_status" / f"{side}_arm.txt"
            arm_state_data = []
            with open(arm_state_file, 'r') as f:
                for line in f:
                    values = [float(x) for x in line.strip().split()]
                    arm_state_data.append(values)
            pose_data_list = None

        return hand_state_data, ee_action_data, pose_data_list, arm_state_data


    def _load_vitra_actions(self, sample: Dict, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load VITRA dataset actions and state. Also handles real_robot with annotation offsets.

        Supports two modes:
        - Joint-based (use_end_pose=False): 7D joint angles + 12D hand = 19D
        - Pose-based (use_end_pose=True): 6D end pose (x, y, z, rot_x, rot_y, rot_z) + 12D hand = 18D
        """
        relative_path = sample['relative_path']
        # Derive recording dir from video_path for correct root resolution
        # (handles extra_roots samples where self.root != actual root)
        recording_dir = Path(sample['video_path']).parent

        # For real_robot, adjust frame_idx by start_frame offset since we're loading a segment
        actual_frame_idx = frame_idx
        # if sample.get('dataset') == 'real_robot':
        #     actual_frame_idx = frame_idx + sample.get('start_frame', 0)

        hand_state_data, ee_action_data, pose_data_list, arm_state_data = self._read_vitra_action_state(recording_dir)

        # Determine episode boundaries (vitra and real_robot both use start_frame/end_frame)
        episode_start = sample.get('start_frame', 0)
        episode_end = sample.get('end_frame', None)

        actions, action_mask, current_state, current_state_mask, total_dim, arm_dim = \
            self._load_general_actions_state(
                hand_state_data, ee_action_data, pose_data_list, arm_state_data,
                actual_frame_idx, episode_start, episode_end,
            )
        return self._process_actions_and_state(
            actions, action_mask, current_state, current_state_mask,
            action_dim=total_dim, state_dim=total_dim,
            has_left_arm=False, has_right_arm=True,
            right_arm_dof=arm_dim,
            mapping_type='xhand',
        )

    def _read_agibot_file(self, h5_file, arm_side: str):
        """Read raw sensor data from an Agibot HDF5 file and convert to VITRA-compatible format.

        Reads end-effector position and orientation (quaternion xyzw) from
        /action/end/position [N,2,3] and /action/end/orientation [N,2,4], plus
        effector (hand) commands from /action/effector/position and
        /state/effector/position.

        Converts quaternion orientation to Euler angles (xyz, radians) to match
        the VITRA pose_data_list format used by _load_general_actions_state.

        Args:
            h5_file: Path to the HDF5 file.
            arm_side: 'left' (index 0) or 'right' (index 1).

        Returns:
            Tuple (hand_states, ee_actions, pose_data_list, arm_states, effector_per_hand).
            Returns (None, None, None, None, None) on error.
        """
        import h5py
        side_idx = 0 if arm_side == 'left' else 1
        try:
            with h5py.File(h5_file, 'r') as hf:
                state_pos = hf['/state/end/position'][:]       # [N, 2, 3]
                state_ori = hf['/state/end/orientation'][:]    # [N, 2, 4] xyzw
                action_eff = hf['/action/effector/position'][:]  # [N, effector_total]
                state_eff  = hf['/state/effector/position'][:]   # [N, effector_total]

            N = state_pos.shape[0]
            effector_total = action_eff.shape[1]
            effector_per_hand = effector_total // 2  # 1 for gripper, 6 for agihand
            eff_slice = slice(0, effector_per_hand) if arm_side == 'left' else slice(effector_per_hand, effector_total)

            pose_data_list = []
            for i in range(N):
                xyz = state_pos[i, side_idx]
                euler_xyz = R.from_quat(state_ori[i, side_idx]).as_euler('xyz', degrees=False)
                pose_data_list.append({
                    'x': float(xyz[0]), 'y': float(xyz[1]), 'z': float(xyz[2]),
                    'rx': float(euler_xyz[0]), 'ry': float(euler_xyz[1]), 'rz': float(euler_xyz[2]),
                })

            hand_states = [state_eff[i, eff_slice].tolist() for i in range(N)]
            ee_actions  = [action_eff[i, eff_slice].tolist() for i in range(N)]
            return hand_states, ee_actions, pose_data_list, None, effector_per_hand

        except Exception as e:
            print(f"Error reading {h5_file}: {e}")
            return None, None, None, None, None

    def _load_agibot_actions(self, sample: Dict, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load AgiBotWorld dataset actions and state.

        Uses end-effector pose (6D: x,y,z,rx,ry,rz) from /action/end/* instead of
        joint angles, with per-arm processing via _load_general_actions_state.
        Mirrors _load_vitra_actions combined with read_agibot_file from calculate_statistics.py.
        """
        task_id = sample['task_id']
        episode_id = sample['episode_id']

        h5_file = self.root / "proprio_stats" / task_id / episode_id / "proprio_stats.h5"

        if not h5_file.exists():
            # Fallback: zeros in end-pose dual-arm gripper format (left 7D + right 7D = 14D)
            fallback_dim = 14
            actions = np.zeros((self.chunk_size, fallback_dim), dtype=np.float32)
            action_mask = np.zeros((self.chunk_size, fallback_dim), dtype=bool)
            current_state = np.zeros(fallback_dim, dtype=np.float32)
            current_state_mask = np.zeros(fallback_dim, dtype=bool)
            return self._process_actions_and_state(
                actions, action_mask, current_state, current_state_mask,
                action_dim=fallback_dim, state_dim=fallback_dim,
                has_left_arm=True, has_right_arm=True,
                left_arm_dof=6, right_arm_dof=6,
                mapping_type='gripper',
            )

        hs_left,  ea_left,  pdl_left,  _, effector_per_hand = self._read_agibot_file(h5_file, 'left')
        hs_right, ea_right, pdl_right, _, _                 = self._read_agibot_file(h5_file, 'right')

        if hs_left is None or hs_right is None:
            fallback_dim = 14
            actions = np.zeros((self.chunk_size, fallback_dim), dtype=np.float32)
            action_mask = np.zeros((self.chunk_size, fallback_dim), dtype=bool)
            current_state = np.zeros(fallback_dim, dtype=np.float32)
            current_state_mask = np.zeros(fallback_dim, dtype=bool)
            return self._process_actions_and_state(
                actions, action_mask, current_state, current_state_mask,
                action_dim=fallback_dim, state_dim=fallback_dim,
                has_left_arm=True, has_right_arm=True,
                left_arm_dof=6, right_arm_dof=6,
                mapping_type='gripper',
            )

        episode_start = sample.get('start_frame', 0)
        episode_end   = sample.get('end_frame', None)

        # Process each arm independently (mirrors _load_vitra_actions per arm)
        left_actions,  left_mask,  left_state,  left_state_mask,  total_dim, arm_dim = \
            self._load_general_actions_state(
                hs_left, ea_left, pdl_left, None,
                frame_idx, episode_start, episode_end,
                ee_dim=effector_per_hand,
            )
        right_actions, right_mask, right_state, right_state_mask, _,         _       = \
            self._load_general_actions_state(
                hs_right, ea_right, pdl_right, None,
                frame_idx, episode_start, episode_end,
                ee_dim=effector_per_hand,
            )

        # Combine left + right arm data: [left | right] concatenated per frame
        combined_actions      = np.concatenate([left_actions,     right_actions],     axis=1)
        combined_mask         = np.concatenate([left_mask,        right_mask],        axis=1)
        combined_state        = np.concatenate([left_state,       right_state])
        combined_state_mask   = np.concatenate([left_state_mask,  right_state_mask])
        combined_dim          = total_dim * 2

        if effector_per_hand == 1:
            mapping_type = 'gripper'
            normalizer = self.gaussian_normalizer_gripper if hasattr(self, 'gaussian_normalizer_gripper') else self.gaussian_normalizer
        elif effector_per_hand == 6:
            mapping_type = 'agihand'
            normalizer = self.gaussian_normalizer_agihand if hasattr(self, 'gaussian_normalizer_agihand') else self.gaussian_normalizer
        else:
            mapping_type = 'gripper'
            normalizer = self.gaussian_normalizer

        return self._process_actions_and_state(
            combined_actions, combined_mask, combined_state, combined_state_mask,
            action_dim=combined_dim, state_dim=combined_dim,
            has_left_arm=True, has_right_arm=True,
            left_arm_dof=arm_dim, right_arm_dof=arm_dim,
            mapping_type=mapping_type,
            normalizer=normalizer,
            use_ee_action=self.use_non_vitra_ee_action,
        )
    
    def _read_molmoact_file(self, parquet_path: Path):
        """Read raw data from a MolmoAct parquet file and convert to VITRA-compatible format.

        Follows the same convention as _read_vitra_action_state and _read_agibot_file:
          - pose_data_list contains N state poses (current arm state per frame).
          - hand_states contains the current gripper state per frame.
          - ee_actions contains the absolute gripper target per frame.

        State format in parquet: [x, y, z, rx, ry, rz, gripper] (7D)
          - xyz in metres; euler angles in radians; gripper is an absolute position.
        Action format in parquet: [x_target, y_target, z_target, rx_target, ry_target, rz_target, d_gripper] (7D)
          - First 6 dims are the absolute arm target pose (next state).
          - Last dim (d_gripper) is a delta: abs_gripper_target = state_gripper + d_gripper.

        Returns:
            hand_states: List of [gripper_state] per frame (N entries).
            ee_actions: List of [abs_gripper_target] per frame (N entries).
            pose_data_list: List of N state pose dicts {x, y, z, rx, ry, rz}.
            arm_states: None (always use end-pose mode for MolmoAct).
            Returns (None, None, None, None) on error.
        """
        import pyarrow.parquet as pq
        try:
            table = pq.read_table(parquet_path)
            N = len(table)
            if N == 0:
                return None, None, None, None

            states_raw = [table['state'][i].as_py() for i in range(N)]
            actions_raw = [table['actions'][i].as_py() for i in range(N)]

            # pose_data_list: N state poses (current arm pose per frame)
            pose_data_list = [
                {'x': float(s[0]), 'y': float(s[1]), 'z': float(s[2]),
                 'rx': float(s[3]), 'ry': float(s[4]), 'rz': float(s[5])}
                for s in states_raw
            ]

            # hand_states: current gripper state per frame
            hand_states = [[float(s[6])] for s in states_raw]

            # ee_actions: absolute gripper target = current gripper + delta gripper
            ee_actions = [[float(s[6]) + float(a[6])] for s, a in zip(states_raw, actions_raw)]

            return hand_states, ee_actions, pose_data_list, None

        except Exception as e:
            print(f"Error reading {parquet_path}: {e}")
            return None, None, None, None

    def _load_molmoact_actions(self, sample: Dict, frame_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load MolmoAct dataset actions and state.

        Uses end-effector pose (6D: x,y,z,rx,ry,rz) from parquet state column, with
        gripper action accumulated from delta (state_gripper + d_gripper).
        Mirrors _load_vitra_actions using _read_molmoact_file + _load_general_actions_state.
        """
        parquet_path = self.root / "train" / "data_new" / sample['chunk_name'] / f"{sample['episode_name']}.parquet"

        hand_states, ee_actions, pose_data_list, arm_states = self._read_molmoact_file(parquet_path)

        if hand_states is None:
            fallback_dim = 7
            actions = np.zeros((self.chunk_size, fallback_dim), dtype=np.float32)
            action_mask = np.zeros((self.chunk_size, fallback_dim), dtype=bool)
            current_state = np.zeros(fallback_dim, dtype=np.float32)
            current_state_mask = np.zeros(fallback_dim, dtype=bool)
            return self._process_actions_and_state(
                actions, action_mask, current_state, current_state_mask,
                action_dim=fallback_dim, state_dim=fallback_dim,
                has_left_arm=False, has_right_arm=True,
                right_arm_dof=6,
                mapping_type='gripper',
            )

        episode_start = sample.get('start_frame', 0)
        episode_end = sample.get('end_frame', None)

        actions, action_mask, current_state, current_state_mask, total_dim, arm_dim = \
            self._load_general_actions_state(
                hand_states, ee_actions, pose_data_list, arm_states,
                frame_idx, episode_start, episode_end,
                ee_dim=1,
            )

        return self._process_actions_and_state(
            actions, action_mask, current_state, current_state_mask,
            action_dim=total_dim, state_dim=total_dim,
            has_left_arm=False, has_right_arm=True,
            right_arm_dof=arm_dim,
            mapping_type='gripper',
            use_ee_action=self.use_non_vitra_ee_action,
        )
    
    def _process_actions_and_state(
        self,
        actions: np.ndarray,
        action_mask: np.ndarray,
        current_state: np.ndarray,
        current_state_mask: np.ndarray,
        action_dim: int,
        state_dim: int,
        has_left_arm: bool,
        has_right_arm: bool,
        left_arm_dof: int = 0,
        right_arm_dof: int = 0,
        mapping_type: str = 'xhand',
        normalizer: Optional[GaussianNormalizer] = None,
        use_ee_action: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Process actions and state to unified dimensions with proper 7-DoF mapping.
        Follows robot_dataset.py pattern: normalize -> extract 7-DoF -> pad -> transfer.
        
        Args:
            mapping_type: Type of robot-to-human mapping ('xhand', 'gripper', 'agihand')
            normalizer: Optional normalizer to use (defaults to self.gaussian_normalizer)
        """
        # Convert to tensors
        actions_tensor = torch.from_numpy(actions).float()
        action_mask_tensor = torch.from_numpy(action_mask).bool()
        current_state_tensor = torch.from_numpy(current_state).float()
        current_state_mask_tensor = torch.from_numpy(current_state_mask).bool()
        
        # Normalize using provided normalizer or default
        if normalizer is None:
            normalizer = self.gaussian_normalizer
        actions_tensor = normalizer.normalize_action(actions_tensor)
        current_state_tensor = normalizer.normalize_state(current_state_tensor)
        
        # Handle 7-DoF extraction: extract first DoF from each 7-DoF arm and store separately
        extra_1d_left = None
        extra_1d_right = None
        extra_1d_left_mask = None
        extra_1d_right_mask = None
        
        left_offset = 0
        right_offset = 0
        
        if has_left_arm and left_arm_dof == 7:
            left_data_dim = action_dim // 2 if has_right_arm else action_dim
            # Extract first DoF
            extra_1d_left = actions_tensor[:, 0]  # [T]
            extra_1d_left_mask = action_mask_tensor[:, 0]
            extra_1d_left_state = current_state_tensor[0]
            extra_1d_left_state_mask = current_state_mask_tensor[0]
            # Remaining data starts from index 1
            actions_tensor = torch.cat([actions_tensor[:, 1:left_data_dim], actions_tensor[:, left_data_dim:]], dim=1)
            action_mask_tensor = torch.cat([action_mask_tensor[:, 1:left_data_dim], action_mask_tensor[:, left_data_dim:]], dim=1)
            current_state_tensor = torch.cat([current_state_tensor[1:left_data_dim], current_state_tensor[left_data_dim:]])
            current_state_mask_tensor = torch.cat([current_state_mask_tensor[1:left_data_dim], current_state_mask_tensor[left_data_dim:]])
            # Update dimensions
            action_dim -= 1
            state_dim -= 1
            right_offset = left_data_dim - 1
        elif has_left_arm:
            left_data_dim = action_dim // 2 if has_right_arm else action_dim
            right_offset = left_data_dim
        
        if has_right_arm and right_arm_dof == 7:
            right_data_dim = action_dim - right_offset
            # Extract first DoF from right arm
            extra_1d_right = actions_tensor[:, right_offset]  # [T]
            extra_1d_right_mask = action_mask_tensor[:, right_offset]
            extra_1d_right_state = current_state_tensor[right_offset]
            extra_1d_right_state_mask = current_state_mask_tensor[right_offset]
            # Remove extracted DoF
            if right_offset > 0:
                actions_tensor = torch.cat([actions_tensor[:, :right_offset], actions_tensor[:, right_offset+1:]], dim=1)
                action_mask_tensor = torch.cat([action_mask_tensor[:, :right_offset], action_mask_tensor[:, right_offset+1:]], dim=1)
                current_state_tensor = torch.cat([current_state_tensor[:right_offset], current_state_tensor[right_offset+1:]])
                current_state_mask_tensor = torch.cat([current_state_mask_tensor[:right_offset], current_state_mask_tensor[right_offset+1:]])
            else:
                actions_tensor = actions_tensor[:, 1:]
                action_mask_tensor = action_mask_tensor[:, 1:]
                current_state_tensor = current_state_tensor[1:]
                current_state_mask_tensor = current_state_mask_tensor[1:]
            # Update dimensions
            action_dim -= 1
            state_dim -= 1
        
        # Now pad to unified robot format (192 for action, 212 for state)
        # Use pad_action and pad_state_robot from robot_dataset.py
        # These functions return tensors directly
        unified_action, unified_action_mask = pad_action(
            actions_tensor,
            action_mask_tensor,
            action_dim,
            192,  # unified action dim
            has_left_arm=has_left_arm
        )
        
        unified_state, unified_state_mask = pad_state_robot(
            current_state_tensor,
            current_state_mask_tensor,
            state_dim,
            212,  # unified state dim
            has_left_arm=has_left_arm
        )
        
        # Add back the extracted 7-DoF extra dimensions to ROBOT_*_EXTRA_1D indices
        if extra_1d_left is not None:
            unified_action[:, ActionFeature.ROBOT_LEFT_EXTRA_1D[0]] = extra_1d_left
            unified_action_mask[:, ActionFeature.ROBOT_LEFT_EXTRA_1D[0]] = extra_1d_left_mask
            unified_state[StateFeature.ROBOT_LEFT_EXTRA_1D[0]] = extra_1d_left_state
            unified_state_mask[StateFeature.ROBOT_LEFT_EXTRA_1D[0]] = extra_1d_left_state_mask
        
        if extra_1d_right is not None:
            unified_action[:, ActionFeature.ROBOT_RIGHT_EXTRA_1D[0]] = extra_1d_right
            unified_action_mask[:, ActionFeature.ROBOT_RIGHT_EXTRA_1D[0]] = extra_1d_right_mask
            unified_state[StateFeature.ROBOT_RIGHT_EXTRA_1D[0]] = extra_1d_right_state
            unified_state_mask[StateFeature.ROBOT_RIGHT_EXTRA_1D[0]] = extra_1d_right_state_mask
        
        # Apply appropriate robot-to-human mapping based on mapping_type
        if mapping_type == 'xhand':
            unified_state, unified_state_mask, unified_action, unified_action_mask = transfer_xhand_to_human(
                unified_state, unified_state_mask,
                unified_action, unified_action_mask
            )
        elif mapping_type == 'gripper':
            unified_state, unified_state_mask, unified_action, unified_action_mask = transfer_gripper_to_human(
                unified_state, unified_state_mask,
                unified_action, unified_action_mask,
                use_ee_action=use_ee_action,
            )
        elif mapping_type == 'agihand':
            unified_state, unified_state_mask, unified_action, unified_action_mask = transfer_agihand_to_human(
                unified_state, unified_state_mask,
                unified_action, unified_action_mask,
                use_ee_action=use_ee_action,
            )
        else:
            raise ValueError(f"Unknown mapping_type: {mapping_type}")
        
        return (
            unified_action.numpy(),
            unified_action_mask.numpy(),
            unified_state.numpy(),
            unified_state_mask.numpy()
        )
    
    def __len__(self) -> int:
        return len(self.frame_indices)

    def _make_dummy_sample(self) -> Dict:
        """Return a zero-filled sample so the batch doesn't crash FSDP.

        All action/state masks are False → zero gradient contribution.
        Labels are set to IGNORE_INDEX (-100) so VLM loss is also zero.
        """
        dummy_pil = Image.new('RGB', (224, 224))
        dummy_text = "<image>Task: dummy. Question: dummy. Answer:"
        model_inputs = self.processor(text=dummy_text, images=dummy_pil, return_tensors="pt")
        prompt_len = model_inputs['input_ids'].shape[1]
        rgb_tensor = model_inputs['pixel_values'].squeeze(0)
        n_ch = 4 if self.sensor_input_all else 2
        return {
            'pixel_values': torch.zeros(n_ch, *rgb_tensor.shape),
            'input_ids': model_inputs['input_ids'].squeeze(0),
            'attention_mask': model_inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor([-100], dtype=model_inputs['input_ids'].dtype),
            'prompt_len': prompt_len,
            'sensor_token': '<None>',
            'debug_info': f'dataset=DUMMY_{self.dataset_name} (all retries failed)',
            'actions': torch.zeros(self.chunk_size, 192),
            'action_masks': torch.zeros(self.chunk_size, 192, dtype=torch.bool),
            'current_state': torch.zeros(212),
            'current_state_mask': torch.zeros(212, dtype=torch.bool),
        }

    def _get_next_video_idx(self, current_idx: int) -> int:
        """
        Find the next index that belongs to a different video using pre-built mapping.
        Maintains the same relative position (portion) within the next video.
        For example, if at 50% through current video, will try 50% through next video.
        This is O(1) thanks to sample_to_frame_range.
        """
        current_sample_idx, current_frame_idx = self.frame_indices[current_idx]
        
        # Get the range of frame indices for current video
        current_start_idx, current_end_idx = self.sample_to_frame_range[current_sample_idx]
        
        # Calculate relative position (portion) within current video
        current_video_length = current_end_idx - current_start_idx
        relative_position = current_frame_idx / max(current_video_length - 1, 1)  # 0.0 to 1.0
        
        # Next video starts right after current video's frames
        next_video_start = current_end_idx % len(self.frame_indices)
        
        # Get next video's sample_idx and range
        next_sample_idx = self.frame_indices[next_video_start][0]
        next_start_idx, next_end_idx = self.sample_to_frame_range[next_sample_idx]
        
        # Apply same relative position to next video
        next_video_length = next_end_idx - next_start_idx
        next_frame_idx = int(relative_position * max(next_video_length - 1, 1))
        
        next_idx = (next_start_idx + next_frame_idx) % len(self.frame_indices)
        
        return next_idx
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single training sample.
        If loading fails, tries next videos (not just next frames) until success.
        This ensures consistent batch sizes across GPUs for FSDP.
        """
        # Try to load current sample and fallback to next videos if it fails
        import time as _time
        _retry_deadline = _time.monotonic() + MAX_RETRY_WALL_SECONDS
        _first_fail_reason = None  # Track first failure for diagnosis
        try_idx = idx
        for attempt in range(self.max_retries):
            # Bail out early if we've spent too long retrying (avoids NCCL timeout)
            if _time.monotonic() > _retry_deadline:
                print(f"[DUMMY_TIMEOUT] dataset={self.dataset_name} idx={idx} after {attempt} attempts, "
                      f"first_fail={_first_fail_reason}", flush=True)
                return self._make_dummy_sample()
            try:
                # Get sample and frame index from precomputed mapping
                sample_idx, frame_idx = self.frame_indices[try_idx]
                sample = self.samples[sample_idx]
                
                # For vitra, real_robot, real_robot_old, and hand_pretrain, adjust frame_idx relative to start_frame
                if sample.get('dataset') in ('vitra', 'real_robot', 'real_robot_old', 'hand_pretrain'):
                    start_frame = sample.get('start_frame', 0)
                    frame_idx = start_frame + frame_idx
                
                # Sample sensor token (for real_robot, it's pre-determined in sample)
                sensor_type = self._sample_sensor_token(sample)
                
                # Get semantic label (category) for sensor type from loaded word probabilities
                if sensor_type == 'none':
                    semantic_label = 'none'
                elif sensor_type in self.word_probs and len(self.word_probs[sensor_type]) > 0:
                    # Sample category from available categories for this sensor
                    categories = list(self.word_probs[sensor_type].keys())
                    semantic_label = random.choice(categories)
                else:
                    # Fallback to default semantic labels if word probs not loaded
                    fallback_labels = {
                        'thermal': ['cold', 'room_temperature', 'hot'],
                        'radar': ['hidden', 'absense'],
                        'audio': ['ringing', 'silence'],
                    }
                    semantic_label = random.choice(fallback_labels.get(sensor_type, ['default']))
                
                # Visual clone augmentation: swap to augmented video paths
                dataset_type = sample.get('dataset', '')
                dataset_supports_aug = dataset_type in ('vitra', 'hand_pretrain')
                use_clone_aug = (
                    self.visual_clone_prob > 0
                    and sensor_type != 'none'
                    and dataset_supports_aug
                    and random.random() < self.visual_clone_prob
                )
                clone_aug_suffix = None
                if use_clone_aug:
                    episode_dir = Path(sample['video_path']).parent
                    clone_aug_suffix = self._get_clone_aug_suffix(episode_dir)
                
                # Load RGB frame (use augmented video if clone aug is active)
                if clone_aug_suffix is not None:
                    aug_rgb_path = str(Path(sample['video_path']).parent / f"video{clone_aug_suffix}.mp4")
                    rgb_frame = self._load_rgb_frame(aug_rgb_path, frame_idx)
                    if rgb_frame is None:
                        # Fallback to original RGB
                        clone_aug_suffix = None
                        rgb_frame = self._load_rgb_frame(sample['video_path'], frame_idx)
                else:
                    rgb_frame = self._load_rgb_frame(sample['video_path'], frame_idx)
                if rgb_frame is None:
                    fail_reason = f"RGB_FAIL video={sample['video_path']} frame={frame_idx} clone_aug={clone_aug_suffix}"
                    if _first_fail_reason is None:
                        _first_fail_reason = fail_reason
                    if attempt < self.max_retries - 1:
                        print(f"[RGB_FAIL] dataset={sample.get('dataset','?')} path={sample.get('relative_path','?')} "
                              f"frame={frame_idx} clone_aug={clone_aug_suffix} attempt={attempt+1}/{self.max_retries}", flush=True)
                        try_idx = self._get_next_video_idx(try_idx)
                        continue
                    else:
                        raise RuntimeError(fail_reason)
        
                # Get sensor video path and load sensor overlay frame(s)
                if self.sensor_input_all:
                    # Load all three sensors (radar uses images_masked_first_frame when radar_fix_start_frame is enabled)
                    sensor_video_paths = self._get_sensor_video_path(sample, sensor_type, semantic_label)  # Returns dict
                    # Apply clone aug suffix to sensor paths
                    if clone_aug_suffix is not None and isinstance(sensor_video_paths, dict):
                        sensor_video_paths = {
                            k: (Path(str(v).replace('.mp4', f'{clone_aug_suffix}.mp4')) if v is not None else None)
                            for k, v in sensor_video_paths.items()
                        }
                    sensor_frames = self._load_sensor_frame(sensor_video_paths, frame_idx, sample)  # Returns dict
                    
                    # Check if all sensor frames loaded successfully
                    if any(frame is None for frame in sensor_frames.values()):
                        if attempt < self.max_retries - 1:
                            logger.warning(f"Failed to load some sensor frames, trying next video (attempt {attempt + 1}/{self.max_retries})")
                            try_idx = self._get_next_video_idx(try_idx)
                            continue
                        else:
                            raise RuntimeError(f"Failed to load valid sample after {self.max_retries} attempts starting from index {idx}")
                elif sensor_type == 'none':
                    sensor_frames = None
                else:
                    # Single sensor mode
                    sensor_video_path = self._get_sensor_video_path(sample, sensor_type, semantic_label)
                    # Apply clone aug suffix to sensor path
                    if clone_aug_suffix is not None and sensor_video_path is not None:
                        sensor_video_path = Path(str(sensor_video_path).replace('.mp4', f'{clone_aug_suffix}.mp4'))
                    sensor_frames = self._load_sensor_frame(sensor_video_path, frame_idx, sample)
                    if sensor_frames is None:
                        exists = sensor_video_path.exists() if sensor_video_path else False
                        fail_reason = (f"SENSOR_FAIL sensor={sensor_type} label={semantic_label} "
                                       f"clone_aug={clone_aug_suffix} path={sensor_video_path} exists={exists}")
                        if _first_fail_reason is None:
                            _first_fail_reason = fail_reason
                        print(f"[SENSOR_FAIL] dataset={sample.get('dataset','?')} path={sample.get('relative_path','?')} "
                              f"{fail_reason} attempt={attempt+1}/{self.max_retries}", flush=True)
                        if attempt < self.max_retries - 1:
                            try_idx = self._get_next_video_idx(try_idx)
                            continue
                        else:
                            raise RuntimeError(fail_reason)
                
                # Convert RGB numpy to PIL for processor
                rgb_pil = Image.fromarray(rgb_frame)
                
                # Process RGB through processor to get correct size for vision encoder
                # PaliGemma processor requires text with <image> token when processing images
                rgb_processed = self.processor(text="<image>", images=rgb_pil, return_tensors="pt")
                rgb_tensor = rgb_processed['pixel_values'].squeeze(0)  # [C, H, W]
                
                # Randomly drop RGB tensor if conditions are met
                should_drop_rgb = (
                    not self.RGB_only and 
                    sensor_frames is not None and 
                    random.random() < self.random_drop_RGB
                )
                if should_drop_rgb:
                    rgb_tensor = torch.zeros_like(rgb_tensor)
                
                # Process sensor frame(s) similarly
                if self.RGB_only:
                    # RGB only mode - use zero tensors for all sensors
                    if self.sensor_input_all:
                        # Stack all zero sensor tensors: [4, C, H, W] = [RGB, thermal, radar, acoustic]
                        sensor_tensors = [torch.zeros_like(rgb_tensor) for _ in ['thermal', 'radar', 'acoustic']]
                        pixel_values = torch.stack([rgb_tensor] + sensor_tensors, dim=0)  # [4, C, H, W]
                    else:
                        # Single sensor mode - use zero tensor
                        sensor_tensor = torch.zeros_like(rgb_tensor)
                        pixel_values = torch.stack([rgb_tensor, sensor_tensor], dim=0)  # [2, C, H, W]
                elif self.sensor_input_all:
                    # Stack all sensor tensors: [4, C, H, W] = [RGB, thermal, radar, acoustic]
                    sensor_tensors = []
                    for sensor_name in ['thermal', 'radar', 'acoustic']:
                        if sensor_frames[sensor_name] is not None:
                            sensor_pil = Image.fromarray(sensor_frames[sensor_name])
                            sensor_processed = self.processor(text="<image>", images=sensor_pil, return_tensors="pt")
                            sensor_tensors.append(sensor_processed['pixel_values'].squeeze(0))
                        else:
                            # Use zero tensor if sensor frame failed to load
                            sensor_tensors.append(torch.zeros_like(rgb_tensor))
                    pixel_values = torch.stack([rgb_tensor] + sensor_tensors, dim=0)  # [4, C, H, W]
                elif sensor_frames is None:
                    # None sensor - use zero tensor
                    sensor_tensor = torch.zeros_like(rgb_tensor)
                    pixel_values = torch.stack([rgb_tensor, sensor_tensor], dim=0)  # [2, C, H, W]
                else:
                    # Single sensor mode
                    sensor_pil = Image.fromarray(sensor_frames)
                    sensor_processed = self.processor(text="<image>", images=sensor_pil, return_tensors="pt")
                    sensor_tensor = sensor_processed['pixel_values'].squeeze(0)  # [C, H, W]
                    pixel_values = torch.stack([rgb_tensor, sensor_tensor], dim=0)  # [2, C, H, W]
                
                # Load actions and state
                actions, action_masks, current_state, current_state_mask = self._load_actions_and_state(sample, frame_idx)
                
                # Process instruction with placeholder replacement using sensor type and semantic label
                instruction = self._replace_placeholders(sample['instruction'], sensor_type, semantic_label)
                
                # Text generalization: randomly replace object noun with "item"
                if (self.text_generalization_prob > 0 
                    and sensor_type != 'none'
                    and random.random() < self.text_generalization_prob):
                    instruction = self._generalize_object_name(instruction)
                
                # Add task suffix for VLM part (matching training format)
                # Use single <image> token for Stage 1 (RGB only)
                # Stage 2 will add extra <image> token for dual images
                instruction = instruction.rstrip()
                if not instruction.endswith("."):
                    instruction += "."
                task_desc = "Tell and segment the target object" if self.train_seg_token else "Tell the target object"
                full_instruction = "<image>Task: " + instruction + f" Question: Choose from <None>, <thermal>, <radar>, <audio> for task. {task_desc}. Answer:"
                
                # Create ground truth text (sensor token + object description)
                sensor_token_str = f"<{sensor_type}>" if sensor_type != "none" else "<None>"
                if self.train_seg_token:
                    seg_token_str = self._get_seg_token_str(
                        sample['video_path'], frame_idx, sample.get('dataset', '')
                    )
                    ground_truth_text = f"{sensor_token_str}. {sample['primary_object']}{seg_token_str}"
                else:
                    ground_truth_text = f"{sensor_token_str}. {sample['primary_object']}"
                
                # Process with tokenizer - disable add_eos_token for prompt
                original_add_eos = self.processor.tokenizer.add_eos_token
                self.processor.tokenizer.add_eos_token = False
                
                # Process prompt with processor (processor handles image normalization)
                model_inputs = self.processor(
                    text=full_instruction,
                    images=[rgb_pil],
                    return_tensors="pt"
                )
                
                # Tokenize output for labels (with EOS token)
                output_tokens = self.processor.tokenizer(
                    ground_truth_text,
                    return_tensors="pt",
                    add_special_tokens=False
                )
                
                # Manually append EOS token
                eos_id = self.processor.tokenizer.eos_token_id
                eos_tensor = torch.tensor([[eos_id]], dtype=output_tokens['input_ids'].dtype)
                output_tokens['input_ids'] = torch.cat([output_tokens['input_ids'], eos_tensor], dim=1)
                
                # Restore original setting
                self.processor.tokenizer.add_eos_token = original_add_eos
            
                # Get prompt length for label alignment
                prompt_len = model_inputs['input_ids'].shape[1]
                
                # Build augmentation debug info string
                aug_suffix_str = clone_aug_suffix if clone_aug_suffix else 'none'
                rgb_src = f'video{clone_aug_suffix}.mp4' if clone_aug_suffix else 'video.mp4'
                text_gen = 'yes' if ('item.' in instruction or 'item ' in instruction) else 'no'
                # Resolve sensor video name for debug logging
                if sensor_type == 'none':
                    sensor_vid_name = 'none'
                elif self.sensor_input_all and isinstance(sensor_video_paths, dict):
                    sensor_vid_name = ','.join(str(Path(v).name) for v in sensor_video_paths.values() if v)
                else:
                    sensor_vid_name = str(Path(sensor_video_path).name) if sensor_video_path else 'none'
                debug_info = (f"dataset={sample.get('dataset','?')} path={sample.get('relative_path','?')} "
                              f"sensor={sensor_type} "
                              f"label={semantic_label} clone_aug={aug_suffix_str} "
                              f"rgb={rgb_src} sensor_vid={sensor_vid_name} text_gen={text_gen} "
                              f"instruction={instruction[:80]}")
                
                # Successfully loaded sample, return it
                return {
                    'pixel_values': pixel_values,  # [2, C, H, W]
                    'input_ids': model_inputs['input_ids'].squeeze(0),
                    'attention_mask': model_inputs['attention_mask'].squeeze(0),
                    'labels': output_tokens['input_ids'].squeeze(0),
                    'prompt_len': prompt_len,
                    'sensor_token': sensor_token_str,
                    'debug_info': debug_info,
                    'actions': torch.from_numpy(actions),  # [chunk_size, 192]
                    'action_masks': torch.from_numpy(action_masks),  # [chunk_size, 192]
                    'current_state': torch.from_numpy(current_state),  # [212]
                    'current_state_mask': torch.from_numpy(current_state_mask),  # [212]
                }
                
            except Exception as e:
                if _first_fail_reason is None:
                    _first_fail_reason = f"EXCEPTION: {e}"
                if attempt < self.max_retries - 1:
                    print(f"[RETRY] dataset={self.dataset_name} idx={try_idx} attempt={attempt+1}/{self.max_retries}: {e}", flush=True)
                    try_idx = self._get_next_video_idx(try_idx)
                    continue
                else:
                    # Exhausted all retries — return dummy sample instead of crashing
                    # the DataLoader worker (which deadlocks multi-GPU FSDP).
                    print(f"[DUMMY] dataset={self.dataset_name} idx={idx} first_fail={_first_fail_reason} last_err={e}", flush=True)
                    return self._make_dummy_sample()
