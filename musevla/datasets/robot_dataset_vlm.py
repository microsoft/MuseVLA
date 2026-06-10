"""
robot_dataset_vlm.py

Dataset implementation for VLM-only finetuning on robot datasets.
Loads RGB images, instructions, and metadata for sensor token prediction.
"""

import json
import os
import random
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RoboDatasetVLMCore(Dataset):
    """
    Core dataset class for VLM-only finetuning on robot manipulation data.
    
    Loads first frame from videos, parses metadata (extra_sensor.json/jsonl),
    and generates samples for sensor token prediction task.
    
    Supports four robot datasets:
    - VITRA: all_metadata.json
    - MolmoAct: meta/extra_sensor.json
    - InternA1: meta/extra_sensor.json per task folder
    - AgiBotWorld: task_info/extra_sensor.jsonl
    """
    
    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        sensor_token_prob: Dict[str, float],
        placeholder_replacements_path: Optional[str] = None,
        placeholder_text: str = "<placeholder>",
        processor=None,
        video_sample_frame_number: int = 1,
        seed: int = 42,
        training_frame_count: Optional[int] = None,
        max_samples: Optional[int] = None
    ):
        """
        Initialize VLM dataset.
        
        Args:
            dataset_name: Name of dataset (vitra/molmoact_household/molmoact_tabletop/interna1/agibot)
            root_dir: Root directory containing the dataset
            sensor_token_prob: Probability dict for sampling sensor tokens
            placeholder_replacements_path: Path to folder containing word probability JSON files
            placeholder_text: Placeholder string in instructions
            processor: Text/image processor with tokenizer
            video_sample_frame_number: Number of frames to sample from each video
            seed: Random seed for reproducibility
            training_frame_count: Number of frames used during training (to skip in evaluation)
            max_samples: Maximum number of video samples to load (None for all). Limits loading early for efficiency.
        """
        self.dataset_name = dataset_name
        self.root = Path(root_dir)
        self.sensor_token_prob = sensor_token_prob
        self.placeholder_replacements_path = placeholder_replacements_path
        self.placeholder_text = placeholder_text
        self.processor = processor
        self.video_sample_frame_number = max(1, video_sample_frame_number)
        self.training_frame_count = training_frame_count
        self.max_samples = max_samples
        self.seed = seed  # Store seed for deterministic sampling
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Normalize probabilities
        total_prob = sum(self.sensor_token_prob.values())
        self.sensor_token_prob = {k: v/total_prob for k, v in self.sensor_token_prob.items()}
        
        # Load word replacement probabilities
        self.word_probs = self._load_word_probabilities()
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Normalize probabilities
        total_prob = sum(self.sensor_token_prob.values())
        self.sensor_token_prob = {k: v/total_prob for k, v in self.sensor_token_prob.items()}
        
        # Parse dataset structure and load metadata
        # Pass max_samples to enable early termination during loading
        video_samples = self._load_samples(self.max_samples)
        
        # Expand samples: create one entry per frame
        self.samples = []
        for video_sample in video_samples:
            for frame_idx in range(self.video_sample_frame_number):
                self.samples.append({
                    **video_sample,
                    'frame_idx': frame_idx
                })
        
        # print(f"Loaded {len(video_samples)} videos ({len(self.samples)} frame samples) from {dataset_name}")
    
    def _load_word_probabilities(self) -> Dict[str, Dict[str, float]]:
        """
        Load word probabilities from JSON files for each sensor type.
        
        Returns:
            Dictionary mapping sensor type to word:probability dict
        """
        word_probs = {}
        
        if not self.placeholder_replacements_path:
            # If no path provided, return empty dict
            return word_probs
        
        replacements_path = Path(self.placeholder_replacements_path)
        if not replacements_path.exists():
            print(f"Warning: Word dict path not found: {replacements_path}")
            return word_probs
        
        # Map sensor token names to file prefixes
        sensor_file_map = {
            'thermal': 'thermal_word_probs_en.json',
            'audio': 'acoustic_word_probs_en.json',
            'radar': 'mmwave_word_probs_en.json',
        }
        
        for sensor_type, filename in sensor_file_map.items():
            file_path = replacements_path / filename
            if not file_path.exists():
                print(f"Warning: Word prob file not found: {file_path}")
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Flatten the nested structure: ignore the first level key and second category
                # Just collect all words and their probabilities
                words_and_probs = {}
                for sensor_key, categories in data.items():
                    # categories is a dict of category_name -> {word: prob}
                    for category_name, word_dict in categories.items():
                        # Merge all words from all categories
                        words_and_probs.update(word_dict)
                
                # Normalize probabilities
                total_prob = sum(words_and_probs.values())
                if total_prob > 0:
                    words_and_probs = {word: prob/total_prob for word, prob in words_and_probs.items()}
                
                word_probs[sensor_type] = words_and_probs
                # print(f"Loaded {len(words_and_probs)} words for {sensor_type} sensor")
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        return word_probs
    
    def _load_samples(self, max_samples: Optional[int] = None) -> List[Dict]:
        """Load all samples based on dataset structure."""
        if self.dataset_name == "vitra":
            return self._load_vitra_samples(max_samples)
        elif self.dataset_name in ["real_robot", "real_robot_acoustic", "real_robot_radar", "real_robot_thermal"]:
            return self._load_real_robot_samples(max_samples)
        elif self.dataset_name.startswith("molmoact_"):
            return self._load_molmoact_samples(max_samples)
        elif self.dataset_name == "interna1":
            return self._load_interna1_samples(max_samples)
        elif self.dataset_name == "agibot":
            return self._load_agibot_samples(max_samples)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_vitra_samples(self, max_samples: Optional[int] = None) -> List[Dict]:
        """Load VITRA dataset samples."""
        metadata_file = self.root / "all_metadata_dist.json"
        if not metadata_file.exists():
            print(f"Warning: Metadata file not found: {metadata_file}")
            return []
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata_list = json.load(f)
        
        # Stride-select for diverse sampling when max_samples limits the count
        if max_samples is not None and max_samples < len(metadata_list):
            stride = max(1, len(metadata_list) // max_samples)
            metadata_list = metadata_list[::stride]
        
        samples = []
        for item in metadata_list:
            # Early termination if max_samples reached
            if max_samples is not None and len(samples) >= max_samples:
                break
                
            video_path = self.root / item['relative_path'] / "video.mp4"
            if not video_path.exists():
                continue
            
            rephrased = item.get('rephrased_instruction', '')
            if self.placeholder_text not in rephrased:
                # If no placeholder, use original instruction
                rephrased = item.get('instruction', '')
            
            samples.append({
                'video_path': str(video_path),
                'instruction': rephrased,
                'primary_object': item.get('primary_object', ''),
                'all_objects': item.get('all_objects', ''),
                'dataset': 'vitra'
            })
        
        return samples
    
    def _load_real_robot_samples(self, max_samples: Optional[int] = None) -> List[Dict]:
        """Load real robot samples from Muse_Robot_Dataset structure.
        
        Each annotation in a video creates a separate sample with:
        - sensor token pre-determined by data folder type
        - primary_object extracted from instruction
        
        Supports filtering by dataset_name:
        - 'real_robot': Load all sensor folders
        - 'real_robot_acoustic': Load only acoustic_data
        - 'real_robot_radar': Load only radar_data
        - 'real_robot_thermal': Load only thermal_data
        """
        samples = []
        
        # Determine which sensor folders to load based on dataset_name
        if self.dataset_name == 'real_robot_acoustic':
            sensor_folders = ['acoustic_data']
        elif self.dataset_name == 'real_robot_radar':
            sensor_folders = ['radar_data']
        elif self.dataset_name == 'real_robot_thermal':
            sensor_folders = ['thermal_data']
        else:  # 'real_robot' - load all
            sensor_folders = ['acoustic_data', 'radar_data', 'thermal_data']
        
        # Map folder names to sensor tokens
        folder_to_sensor = {
            'acoustic_data': 'audio',
            'radar_data': 'radar',
            'thermal_data': 'thermal'
        }
        
        # Distribute max_samples equally across sensor folders
        per_sensor_max = None
        if max_samples is not None:
            per_sensor_max = max(1, max_samples // len(sensor_folders))
        
        for data_folder_name in sensor_folders:
            data_folder = self.root / data_folder_name
            if not data_folder.exists():
                continue
            
            label_folder = self.root / data_folder_name.replace('_data', '_labels')
            
            sensor_count = 0  # Track samples loaded for this sensor
            episode_folders = sorted([d for d in data_folder.iterdir() if d.is_dir()])
            # Stride-select for diverse episode sampling when per-sensor cap limits the count
            if per_sensor_max is not None and per_sensor_max < len(episode_folders):
                stride = max(1, len(episode_folders) // per_sensor_max)
                episode_folders = episode_folders[::stride]
            for episode_folder in episode_folders:
                if per_sensor_max is not None and sensor_count >= per_sensor_max:
                    break
                
                episode_name = episode_folder.name
                
                # Check label file for annotations
                label_file = label_folder / f"{episode_name}.json"
                if not label_file.exists():
                    continue
                
                try:
                    with open(label_file, 'r', encoding='utf-8') as f:
                        label_data = json.load(f)
                    
                    if label_data.get('problem', {}).get('abolished', False):
                        continue
                    
                    # Get annotations
                    annotations = label_data.get('annotations', [])
                    if not annotations:
                        continue
                    
                except Exception as e:
                    continue
                
                # Check if video.mp4 exists
                video_path = episode_folder / "video.mp4"
                if not video_path.exists():
                    continue
                
                # Get total frame count from label file
                total_frames_label = label_data.get('frame_num_total', None)

                # Create a sample for each annotation (following robot_dataset_vlm_vla.py logic)
                for ann_idx, annotation in enumerate(annotations):
                    if per_sensor_max is not None and sensor_count >= per_sensor_max:
                        break
                    
                    instruction = annotation.get('instruction', '')
                    start_frame = annotation.get('start', 0)
                    end_frame   = annotation.get('end', total_frames_label - 1 if total_frames_label else None)

                    # Determine primary_object based on data_type and instruction
                    instruction_lower = instruction.lower()
                    if data_folder_name == 'acoustic_data':
                        if 'towel' in instruction_lower:
                            primary_object = 'towels'
                        elif 'cloth' in instruction_lower:
                            primary_object = 'clothes'
                        else:
                            primary_object = 'towels'
                    elif data_folder_name == 'thermal_data':
                        primary_object = 'bottles'
                    elif data_folder_name == 'radar_data':
                        primary_object = 'boxes'
                    else:
                        primary_object = 'none'
                    
                    # Mirror robot_dataset_vlm_vla.py: sensor only for first annotation
                    # when the task is not a placement task
                    if ann_idx == 0 and 'place' not in instruction_lower:
                        # First annotation and not a placement task: determine from data_type
                        if data_folder_name == 'acoustic_data':
                            sensor_token = 'audio'
                        elif data_folder_name == 'thermal_data':
                            sensor_token = 'thermal'
                        elif data_folder_name == 'radar_data':
                            sensor_token = 'radar'
                        else:
                            sensor_token = 'none'
                    else:
                        # Second+ annotation or placement task: no sensor
                        sensor_token = 'none'
                    
                    samples.append({
                        'video_path':    str(video_path),
                        'instruction':   instruction,
                        'primary_object': primary_object,
                        'all_objects':   primary_object,
                        'dataset':       self.dataset_name,
                        'sensor_token':  sensor_token,
                        'start_frame':   start_frame,
                        'end_frame':     end_frame,
                    })
                    sensor_count += 1
        
        return samples
    
    def _load_molmoact_samples(self, max_samples: Optional[int] = None) -> List[Dict]:
        """Load MolmoAct dataset samples."""
        # Determine split (household or tabletop)
        split = self.dataset_name.split('_')[1]  # household or tabletop
        
        samples = []
        for split_dir in ['train']:
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
                print(f"Warning: tasks.jsonl not found in {meta_dir}")
                continue
            
            tasks = {}
            with open(tasks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    task_data = json.loads(line.strip())
                    tasks[task_data['task_index']] = task_data['task']
            
            # Create reverse mapping: task_description -> task_index
            task_desc_to_idx = {desc: idx for idx, desc in tasks.items()}
            
            # Load episodes.jsonl to map episodes to tasks
            episodes_file = meta_dir / "episodes.jsonl"
            if not episodes_file.exists():
                print(f"Warning: episodes.jsonl not found in {meta_dir}")
                continue
            
            # Build mapping of task_index -> list of episode indices
            task_to_episodes = {}
            with open(episodes_file, 'r', encoding='utf-8') as f:
                for line in f:
                    episode_data = json.loads(line.strip())
                    episode_idx = episode_data['episode_index']
                    task_list = episode_data['tasks']
                    
                    if task_list:
                        task_desc = task_list[0]  # First task in list
                        task_idx = task_desc_to_idx.get(task_desc)
                        
                        if task_idx is not None:
                            if task_idx not in task_to_episodes:
                                task_to_episodes[task_idx] = []
                            task_to_episodes[task_idx].append(episode_idx)
            
            # Now iterate through metadata and create samples for each episode
            for task_key, task_meta in metadata_dict.items():
                # Early termination if max_samples reached
                if max_samples is not None and len(samples) >= max_samples:
                    break
                    
                task_index = task_meta.get('task_index')
                rephrased = task_meta.get('rephrased_instruction', '')
                
                # Get all episode indices for this task
                episode_indices = task_to_episodes.get(task_index, [])
                
                for episode_idx in episode_indices:
                    # Early termination if max_samples reached
                    if max_samples is not None and len(samples) >= max_samples:
                        break
                        
                    # Build video path: train/videos/chunk-XXX/observations.rgb/episode_YYYYYY.mp4
                    chunk_num = episode_idx // 1000
                    chunk_name = f"chunk-{chunk_num:03d}"
                    episode_name = f"episode_{episode_idx:06d}.mp4"
                    
                    video_path = self.root / split_dir / "videos" / chunk_name / "observations.rgb" / episode_name
                    
                    if not video_path.exists():
                        continue
                    
                    samples.append({
                        'video_path': str(video_path),
                        'instruction': rephrased,
                        'primary_object': task_meta.get('primary_object', ''),
                        'all_objects': task_meta.get('all_objects', ''),
                        'dataset': f'molmoact_{split}'
                    })
        
        return samples
    
    def _load_interna1_samples(self, max_samples: Optional[int] = None) -> List[Dict]:
        """Load InternA1 dataset samples."""
        samples = []
        
        # Iterate through task folders
        for task_folder in self.root.iterdir():
            # Early termination if max_samples reached
            if max_samples is not None and len(samples) >= max_samples:
                break
                
            if not task_folder.is_dir() or task_folder.name.endswith('.tar.gz'):
                continue
            
            metadata_file = task_folder / "set_0" /"meta" / "extra_sensor.json"
            if not metadata_file.exists():
                continue
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            
            # Check for RGB videos
            video_dir = task_folder / "set_0" / "videos" / "chunk-000" / "observation.images.head"
            if not video_dir.exists():
                continue
            
            for video_file in video_dir.glob("*.mp4"):
                # Early termination if max_samples reached
                if max_samples is not None and len(samples) >= max_samples:
                    break
                    
                # Only use h264 converted versions (skip original AV1 videos)
                if not video_file.stem.endswith('_h264'):
                    continue
                
                # Remove _h264 suffix to get original video name for metadata lookup
                video_name = video_file.stem.replace('_h264', '')
                if video_name in metadata_dict:
                    video_meta = metadata_dict[video_name]
                    samples.append({
                        'video_path': str(video_file),
                        'instruction': video_meta.get('rephrased_instruction', ''),
                        'primary_object': video_meta.get('primary_object', ''),
                        'all_objects': video_meta.get('all_objects', ''),
                        'dataset': 'interna1'
                    })
        
        return samples
    
    def _load_agibot_samples(self, max_samples: Optional[int] = None) -> List[Dict]:
        """Load AgiBotWorld dataset samples (JSONL format)."""
        metadata_file = self.root / "task_info" / "extra_sensor.jsonl"
        if not metadata_file.exists():
            print(f"Warning: Metadata file not found: {metadata_file}")
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
                
                samples.append({
                    'video_path': str(video_path),
                    'instruction': item.get('rephrased_instruction', item.get('original_instruction', '')),
                    'primary_object': item.get('primary_object', ''),
                    'all_objects': item.get('all_objects', ''),
                    'dataset': 'agibot'
                })
        
        return samples
    
    def _sample_sensor_token(self) -> str:
        """Sample a sensor token based on configured probabilities."""
        tokens = list(self.sensor_token_prob.keys())
        probs = [self.sensor_token_prob[t] for t in tokens]
        return random.choices(tokens, weights=probs)[0]
    
    def _process_instruction(self, instruction: str, sensor_token: str) -> str:
        """
        Process instruction by replacing placeholder with appropriate text.
        
        Args:
            instruction: Original instruction with placeholder
            sensor_token: Sampled sensor token (none/thermal/radar/audio)
        
        Returns:
            Processed instruction text
        """
        if sensor_token == "none":
            # Remove placeholder entirely
            return instruction.replace(self.placeholder_text, "").strip()
        else:
            # Sample replacement word from probabilities
            if sensor_token in self.word_probs and len(self.word_probs[sensor_token]) > 0:
                words = list(self.word_probs[sensor_token].keys())
                probs = [self.word_probs[sensor_token][w] for w in words]
                replacement = random.choices(words, weights=probs)[0]
            else:
                # Fallback to empty if no words available
                print(f"Warning: No replacement words for sensor {sensor_token}")
                replacement = ""
            
            return instruction.replace(self.placeholder_text, replacement).strip()
    
    def _process_instruction_deterministic(self, instruction: str, sensor_token: str, rng: random.Random) -> str:
        """
        Process instruction by replacing placeholder with appropriate text, using deterministic RNG.
        
        Args:
            instruction: Original instruction with placeholder
            sensor_token: Sampled sensor token (none/thermal/radar/audio)
            rng: Random number generator for deterministic sampling
        
        Returns:
            Processed instruction text
        """
        if sensor_token == "none":
            # Remove placeholder entirely
            return instruction.replace(self.placeholder_text + " ", "").strip()
        else:
            # Sample replacement word from probabilities using provided RNG
            if sensor_token in self.word_probs and len(self.word_probs[sensor_token]) > 0:
                words = list(self.word_probs[sensor_token].keys())
                probs = [self.word_probs[sensor_token][w] for w in words]
                replacement = rng.choices(words, weights=probs)[0]
            else:
                # Fallback to empty if no words available
                print(f"Warning: No replacement words for sensor {sensor_token}")
                replacement = ""
            
            return instruction.replace(self.placeholder_text, replacement).strip()
    
    def _extract_frame(self, video_path: str, frame_idx: int,
                       start_frame: int = 0, end_frame: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Extract a single frame from video at the specified sample index.
        
        Args:
            video_path: Path to video file
            frame_idx: Frame sample index (0 to video_sample_frame_number-1)
            start_frame: First valid frame in the annotation window (inclusive)
            end_frame: Last valid frame in the annotation window (inclusive);
                       None means use the last frame of the video
        
        Returns:
            RGB numpy array (H, W, 3) or None if failed
        """
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                cap.release()
                return None

            # Clamp annotation window to actual video length
            clip_start = max(0, start_frame)
            clip_end   = min(total_frames - 1, end_frame if end_frame is not None else total_frames - 1)
            if clip_start > clip_end:
                clip_start = 0
                clip_end   = total_frames - 1
            
            # Calculate frame indices to sample
            if self.training_frame_count is not None:
                # Skip frames used during training.
                # Training frames are sampled evenly within the annotation window.
                if self.training_frame_count == 1:
                    training_frame_indices = [clip_start]
                elif self.training_frame_count == 2:
                    training_frame_indices = [clip_start, clip_end]
                else:
                    training_frame_indices = list(np.linspace(clip_start, clip_end, self.training_frame_count, dtype=int))
                
                # Available frames are within the annotation window, minus training frames
                all_frame_indices = list(range(clip_start, clip_end + 1))
                available_frames  = [f for f in all_frame_indices if f not in training_frame_indices]
                
                # If we don't have enough frames, just use what's available
                if len(available_frames) < self.video_sample_frame_number:
                    print(f"Warning: Video {video_path} has insufficient non-training frames. Using available frames.")
                    if len(available_frames) == 0:
                        available_frames = all_frame_indices
                    
                    if len(available_frames) < self.video_sample_frame_number:
                        target_frame_idx = available_frames[frame_idx % len(available_frames)]
                    else:
                        eval_frame_indices = np.linspace(0, len(available_frames) - 1, self.video_sample_frame_number, dtype=int)
                        target_frame_idx = available_frames[eval_frame_indices[frame_idx]]
                else:
                    eval_frame_indices = np.linspace(0, len(available_frames) - 1, self.video_sample_frame_number, dtype=int)
                    target_frame_idx = available_frames[eval_frame_indices[frame_idx]]
            else:
                # Sample evenly within the annotation window
                if self.video_sample_frame_number == 1:
                    target_frame_idx = clip_start
                elif self.video_sample_frame_number == 2:
                    target_frame_idx = clip_start if frame_idx == 0 else clip_end
                else:
                    frame_indices    = np.linspace(clip_start, clip_end, self.video_sample_frame_number, dtype=int)
                    target_frame_idx = frame_indices[frame_idx]
            
            # Read the specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                return None
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame_rgb
            
        except Exception as e:
            print(f"Error extracting frame from {video_path}: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            Dictionary with keys:
            - image: PIL Image
            - instruction: Processed text instruction
            - sensor_token: Sampled sensor token string
            - object_description: Primary object name
            - dataset_name: Source dataset name
        """
        sample = self.samples[idx]
        
        # Extract single frame, respecting annotation window for real_robot samples
        frame = self._extract_frame(
            sample['video_path'],
            sample['frame_idx'],
            start_frame=sample.get('start_frame', 0),
            end_frame=sample.get('end_frame', None),
        )
        if frame is None:
            # Return dummy data if frame extraction fails
            frame = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Convert to PIL Image
        image = Image.fromarray(frame)
        
        # Create deterministic RNG for this sample
        rng = random.Random(self.seed + idx)
        
        # For real_robot, sensor token is pre-determined; for others, sample randomly
        if 'sensor_token' in sample and sample['sensor_token']:
            # Use pre-determined sensor token (for real_robot dataset)
            sensor_token = sample['sensor_token']
        else:
            # Sample sensor token deterministically based on sample index
            # This ensures same sample always gets same sensor token across runs
            tokens = list(self.sensor_token_prob.keys())
            probs = [self.sensor_token_prob[t] for t in tokens]
            sensor_token = rng.choices(tokens, weights=probs)[0]
        
        # Process instruction with deterministic word sampling
        instruction = sample['instruction']
        is_real_robot = self.dataset_name.startswith('real_robot')
        if is_real_robot:
            # Real robot instructions are already complete from label JSON — use as-is
            processed_instruction = instruction.strip()
        else:
            if not instruction or self.placeholder_text not in instruction:
                # If no placeholder, append it
                instruction = instruction + " " + self.placeholder_text
            processed_instruction = self._process_instruction_deterministic(instruction, sensor_token, rng)
        
        # Get object description
        object_description = sample['primary_object']
        
        return {
            'image': image,
            'instruction': processed_instruction,
            'sensor_token': sensor_token,
            'object_description': object_description,
            'dataset_name': sample['dataset']
        }


class RoboDatasetVLM(Dataset):
    """
    Wrapper dataset for VLM finetuning with processor integration.
    Converts raw samples to model inputs.
    """
    
    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        sensor_token_prob: Dict[str, float],
        placeholder_replacements_path: Optional[str] = None,
        placeholder_text: str = "<placeholder>",
        processor=None,
        augmentation: bool = False,
        flip_augmentation: float = 1.0,
        video_sample_frame_number: int = 1,
        repeat_sensor_token_num: int = 3,
        seed: int = 42,
        train_seg_token: bool = False,
    ):
        self.core_dataset = RoboDatasetVLMCore(
            dataset_name=dataset_name,
            root_dir=root_dir,
            sensor_token_prob=sensor_token_prob,
            placeholder_replacements_path=placeholder_replacements_path,
            placeholder_text=placeholder_text,
            processor=processor,
            video_sample_frame_number=video_sample_frame_number,
            seed=seed
        )
        self.processor = processor
        self.dataset_name = dataset_name
        self.augmentation = augmentation
        self.flip_augmentation = flip_augmentation
        self.repeat_sensor_token_num = repeat_sensor_token_num
        self.train_seg_token = train_seg_token
        
        # Map sensor token strings to special tokens
        self.sensor_token_map = {
            "none": "<None>",
            "thermal": "<thermal>",
            "radar": "<radar>",
            "audio": "<audio>"
        }

        # # Map sensor token strings to simple words
        # self.sensor_token_map = {
        #     "none": "none",
        #     "thermal": "thermal",
        #     "radar": "radar",
        #     "audio": "audio"
        # }
    
    def __len__(self) -> int:
        return len(self.core_dataset)

    def _get_seg_token_str(self, video_path: str, frame_idx: int, dataset_name: str) -> str:
        """Return PaliGemma-style loc+seg token string for a given video frame.

        frame_idx is an absolute video frame index (same as used when reading the RGB video),
        which maps 1-to-1 to the token file since n == total video frames.
        """
        vp = Path(video_path)
        if dataset_name == 'vitra':
            token_path = vp.parent / "video_seg_tokens.npz"
        elif dataset_name == 'agibot':
            token_path = vp.parent.parent / "head_mask_tokens.npz"
        elif dataset_name == 'interna1':
            stem = vp.stem.replace('_h264', '')
            token_path = vp.parent.parent / "observation.images.mask" / f"{stem}_tokens.npz"
        elif dataset_name.startswith('molmoact_'):
            token_path = vp.parent.parent / "observations.mask" / f"{vp.stem}_tokens.npz"
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

    def __getitem__(self, idx: int) -> Dict:
        """
        Get processed sample ready for model input.
        
        Returns:
            Dictionary with tokenized inputs and labels
        """
        sample = self.core_dataset[idx]
        
        image = sample['image']
        
        # Apply augmentation if enabled
        if self.augmentation:
            from musevla.datasets.augment_utils import apply_color_augmentation, contains_color_word
            
            # Convert PIL to numpy for augmentation
            image_np = np.array(image)
            
            # Apply horizontal flip augmentation
            flip_prob = self.flip_augmentation * 0.5  # flip_augmentation controls probability
            if random.random() < flip_prob:
                image_np = np.fliplr(image_np).copy()
            
            # Preserve hue if instruction mentions colors
            preserve_hue = contains_color_word(sample['instruction'])
            
            # Apply color augmentation
            image_np = apply_color_augmentation(
                image_np,
                brightness=0.3,
                contrast=0.3,
                saturation=0.4,
                hue=0.3,
                p=0.8,
                preserve_hue=preserve_hue
            )
            
            # Convert back to PIL
            image = Image.fromarray(image_np)
        
        # Prepare text input: <image> + instruction
        # text = f"<image>{sample['instruction']}. Choose a sensor from none, thermal, radar, acoustic, and also tell the target object."
        # text = f"<image> Task: {sample['instruction']}. Choose from none, thermal, radar, audio. Tell the target object."

        instruction = sample["instruction"].rstrip()
        if not instruction.endswith("."):
            instruction += "."
        task_desc = "Tell and segment the target object" if self.train_seg_token else "Tell the target object"
        text = (
            f"<image>Task: {instruction} Question: Choose from <None>, <thermal>, <radar>, <audio> for task. {task_desc}. Answer:"
        )
        # text = (
        #     f"<image>Task: {instruction} Question: Choose from none, thermal, radar, audio for task. Tell the target object. Answer:"
        # )
        
        # Prepare text output: sensor_token + object_description
        sensor_token_str = self.sensor_token_map[sample['sensor_token']]
        # output_text = f"{sensor_token_str}. {sample['object_description']}"
        if self.train_seg_token:
            raw_sample = self.core_dataset.samples[idx]
            seg_str = self._get_seg_token_str(
                raw_sample['video_path'], raw_sample['frame_idx'], raw_sample['dataset']
            )
            output_text = (f"{sensor_token_str}" + " ") * (self.repeat_sensor_token_num - 1) + f"{sensor_token_str}. {sample['object_description']}{seg_str}"
        else:
            output_text = (f"{sensor_token_str}" + " ") * (self.repeat_sensor_token_num - 1) + f"{sensor_token_str}. {sample['object_description']}"

        
        # IMPORTANT: Disable add_eos_token for prompt to prevent EOS before answer
        original_add_eos = self.processor.tokenizer.add_eos_token
        self.processor.tokenizer.add_eos_token = False
        
        # Process prompt with tokenizer (NO EOS at end)
        model_inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt"
        )
        
        # Tokenize output for labels (WITH EOS token so model learns to stop generating)
        # NOTE: add_eos_token flag doesn't work with add_special_tokens=False
        # So we manually append EOS token
        output_tokens = self.processor.tokenizer(
            output_text,
            return_tensors="pt",
            add_special_tokens=False  # Don't add BOS
        )
        
        # Manually append EOS token (add_eos_token flag doesn't work with add_special_tokens=False)
        eos_id = self.processor.tokenizer.eos_token_id
        eos_tensor = torch.tensor([[eos_id]], dtype=output_tokens['input_ids'].dtype)
        output_tokens['input_ids'] = torch.cat([output_tokens['input_ids'], eos_tensor], dim=1)
        
        # Restore original setting
        self.processor.tokenizer.add_eos_token = original_add_eos
        
        # Get prompt length (before labels) for correct label alignment
        prompt_len = model_inputs['input_ids'].shape[1]
        
        return {
            'pixel_values': model_inputs['pixel_values'].squeeze(0),
            'input_ids': model_inputs['input_ids'].squeeze(0),
            'attention_mask': model_inputs['attention_mask'].squeeze(0),
            'labels': output_tokens['input_ids'].squeeze(0),
            'prompt_len': prompt_len,
            'dataset_name': self.dataset_name,
            'sensor_token': sample['sensor_token'],
        }
