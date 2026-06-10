"""
materialize.py

Dataset and collator materialization for VLA training.
"""

import time
from pathlib import Path
from typing import Tuple

from musevla.datasets.dataset import MultipleWeightedDataset, MultipleDatasetWeightedDistributedBatchSampler
from musevla.datasets.robot_dataset_vlm import RoboDatasetVLM
from musevla.utils.data_utils import PaddedCollatorForHandPrediction, PaddedCollatorForVLM, PaddedCollatorForVLMAndVLA
from torch.utils.data import Dataset




def get_vlm_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    vlm_finetune_config: dict,
    padding_side: str = "right",
    augmentation: bool = False,
    flip_augmentation: float = 1.0,
    shard_num: int = 1,
    shard_index: int = 0,
    seed: int = 42,
    batch_size: int = 32,
    processor=None,
    video_sample_frame_number: int = 1,
) -> Tuple[MultipleWeightedDataset, PaddedCollatorForVLM, MultipleDatasetWeightedDistributedBatchSampler]:
    """
    Create VLM dataset and collator for finetuning.
    
    Args:
        data_root_dir: Root directory containing datasets
        data_mix: Comma-separated list of dataset names
        vlm_finetune_config: VLM finetuning configuration dict
        padding_side: Token padding side ("right" or "left")
        augmentation: Whether to apply data augmentation
        flip_augmentation: Probability of horizontal flip augmentation
        shard_num: Number of distributed shards
        shard_index: Index of current shard
        seed: Random seed for sampling
        batch_size: Batch size per device
        processor: Text processor with tokenizer
    
    Returns:
        Tuple of (dataset, collator, batch_sampler)
    """
    assert 0 <= shard_index < shard_num, "Shard index must be in [0, shard_num)."
    
    # Parse dataset mix and weights
    # Format: "dataset1,dataset2" or "dataset1:0.3,dataset2:0.7"
    dataset_configs = [item.strip() for item in data_mix.split(',')]
    dataset_names = []
    dataset_weights = []
    
    for config in dataset_configs:
        if ':' in config:
            name, weight = config.split(':')
            dataset_names.append(name.strip())
            dataset_weights.append(float(weight.strip()))
        else:
            dataset_names.append(config)
            dataset_weights.append(1.0)  # Default equal weight
    
    # # Normalize weights
    # total_weight = sum(dataset_weights)
    # dataset_weights = [w / total_weight for w in dataset_weights]
    
    # Create individual datasets
    datasets = []
    for dataset_name in dataset_names:
        # Determine dataset root path
        if dataset_name == "vitra":
            dataset_root = Path(data_root_dir) / "VITRA_Robot_Data"
        elif dataset_name == "molmoact_household":
            dataset_root = Path(data_root_dir) / "MolmoAct_Data" / "molmoact_dataset_household"
        elif dataset_name == "molmoact_tabletop":
            dataset_root = Path(data_root_dir) / "MolmoAct_Data" / "molmoact_dataset_tabletop"
        elif dataset_name == "interna1":
            dataset_root = Path(data_root_dir) / "InternA1_Robot_Data" / "real" / "genie1"
        elif dataset_name == "agibot":
            dataset_root = Path(data_root_dir) / "Agibot_Data_Copy"
        else:
            print(f"Warning: Unknown dataset name: {dataset_name}, skipping...")
            continue
        
        if not dataset_root.exists():
            print(f"Warning: Dataset root not found: {dataset_root}, skipping...")
            continue
        
        # Create VLM dataset
        vlm_dataset = RoboDatasetVLM(
            dataset_name=dataset_name,
            root_dir=str(dataset_root),
            sensor_token_prob=vlm_finetune_config.get('sensor_token_prob', {
                'none': 0.25, 'thermal': 0.25, 'radar': 0.25, 'audio': 0.25
            }),
            placeholder_replacements_path=vlm_finetune_config.get('placeholder_replacements_path', None),
            placeholder_text=vlm_finetune_config.get('placeholder_text', '<placeholder>'),
            processor=processor,
            augmentation=augmentation,
            flip_augmentation=flip_augmentation,
            video_sample_frame_number=video_sample_frame_number,
            repeat_sensor_token_num=vlm_finetune_config.get('repeat_sensor_token_num', 3),
            seed=seed + shard_index,  # Different seed per shard
            train_seg_token=vlm_finetune_config.get('train_seg_token', False),
        )
        
        datasets.append(vlm_dataset)
    
    # Create weighted dataset
    if len(datasets) == 0:
        raise ValueError("No valid datasets found!")
    
    # Filter weights to match actual datasets loaded
    final_weights = [dataset_weights[i] for i, name in enumerate(dataset_names) 
                     if any(d.dataset_name == name for d in datasets)]
    
    # # Normalize weights again after filtering
    # total_weight = sum(final_weights)
    # final_weights = [w / total_weight for w in final_weights]
    
    combined_dataset = MultipleWeightedDataset(datasets, weights=final_weights)
    
    # Create batch sampler for weighted sampling
    batch_sampler = MultipleDatasetWeightedDistributedBatchSampler(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        seed=seed,
        num_replicas=shard_num,
        rank=shard_index,
    )
    
    # Create collator for padding
    collator = PaddedCollatorForVLM(
        model_max_length=processor.tokenizer.model_max_length,
        pad_token_id=processor.tokenizer.pad_token_id,
        padding_side=padding_side,
    )
    
    return combined_dataset, collator, batch_sampler

def get_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    padding_side: str = "right",
    augmentation: bool = False,
    shard_num: int = 1,
    shard_index: int = 0,
    seed: int = 42,
    future_action_window_size: int = 0,
    batch_size: int = 32,
    processor=None,
    normalization: bool = True,
    flip_augmentation: float = 1.0,
    set_none_ratio: float = 0.0,
    statistics_type: str = 'ori',
    action_type: str = "angle",
    use_rel: bool = False,
    rel_mode: str = "step",
    clip_len: int = None,
    state_mask_prob: float = 0.1,
) -> Tuple[MultipleWeightedDataset, PaddedCollatorForHandPrediction, MultipleDatasetWeightedDistributedBatchSampler]:
    """
    Create VLA dataset, batch sampler, and collator for training.
    
    Args:
        data_root_dir: Root directory containing datasets
        data_mix: Comma-separated list of dataset names to mix
        padding_side: Token padding side ("right" or "left")
        augmentation: Whether to apply data augmentation
        shard_num: Number of distributed shards
        shard_index: Index of current shard
        seed: Random seed for sampling
        future_action_window_size: Number of future actions to predict
        batch_size: Batch size per device
        processor: Text processor with tokenizer
        normalization: Whether to normalize actions
        flip_augmentation: Probability of flip augmentation
        set_none_ratio: Ratio of setting actions to None
        statistics_type: Type of statistics to use
        action_type: Type of action representation
        use_rel: Whether to use relative actions
        rel_mode: Relative action mode
    
    Returns:
        Tuple of (dataset, collator, batch_sampler)
    """
    assert 0 <= shard_index < shard_num, "Shard index must be in [0, shard_num)."

    multi_dataset = MultipleWeightedDataset.load_datasets(
        data_root_dir,
        data_mix, 
        action_past_window_size=0,
        action_future_window_size=future_action_window_size,
        image_past_window_size=0,
        augmentation=augmentation,
        normalization = normalization,
        processor = processor,
        flip_augmentation = flip_augmentation,
        set_none_ratio = set_none_ratio,
        action_type = action_type, 
        use_rel = use_rel, 
        rel_mode = rel_mode,
        clip_len=clip_len, 
        state_mask_prob=state_mask_prob
    )

    batch_sampler = MultipleDatasetWeightedDistributedBatchSampler(
        multi_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        seed=seed,
        num_replicas=shard_num,
        rank=shard_index,
    )

    # Create collator for padding
    collator = PaddedCollatorForHandPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side=padding_side
    )

    return multi_dataset, collator, batch_sampler


def get_vlm_vla_dataset_and_collator(
    data_root_dir: Path,
    data_mix: str,
    statistics_path: str,
    vlm_vla_config: dict,
    padding_side: str = "right",
    shard_num: int = 1,
    shard_index: int = 0,
    seed: int = 42,
    batch_size: int = 32,
    processor=None,
    action_future_window_size: int = 16,
    action_type: str = "angle",
    vitra_subdir: str = "VITRA_Robot_Data",
    molmoact_subdir: str = "MolmoAct_Data",
) -> Tuple[MultipleWeightedDataset, PaddedCollatorForVLMAndVLA, MultipleDatasetWeightedDistributedBatchSampler]:
    """
    Create VLM-VLA co-training dataset and collator.
    
    Combines VLM sensor token prediction with VLA action prediction.
    
    Args:
        data_root_dir: Root directory containing datasets
        data_mix: Comma-separated list of dataset names
        statistics_path: Path to normalization statistics JSON file
        vlm_vla_config: VLM-VLA configuration dict containing:
            - sensor_token_prob: Dict[str, float] - probabilities for each sensor token
            - background_modes: str - string of background modes (dark/color/grey)
            - placeholder_replacements_path: Optional[str] - path to word replacement JSONs
            - placeholder_text: str - placeholder string in instructions
        padding_side: Token padding side ("right" or "left")
        shard_num: Number of distributed shards
        shard_index: Index of current shard
        seed: Random seed for sampling
        batch_size: Batch size per device
        processor: Text processor with tokenizer
        action_future_window_size: Number of future actions to predict (chunk_size)
        action_type: Type of action representation ('angle' or 'keypoints')
    
    Returns:
        Tuple of (dataset, collator, batch_sampler)
    """
    from musevla.datasets.robot_dataset_vlm_vla import RoboDatasetVLMVLACore
    from musevla.utils.data_utils import PaddedCollatorForVLMAndVLA
    
    assert 0 <= shard_index < shard_num, "Shard index must be in [0, shard_num)."
    
    # Parse dataset mix and weights
    dataset_configs = [item.strip() for item in data_mix.split(',')]
    dataset_names = []
    dataset_weights = []
    
    for config in dataset_configs:
        if ':' in config:
            name, weight = config.split(':')
            dataset_names.append(name.strip())
            dataset_weights.append(float(weight.strip()))
        else:
            dataset_names.append(config)
            dataset_weights.append(1.0)
    
    # Create individual datasets
    datasets = []
    for dataset_name in dataset_names:
        extra_roots = None
        # Determine dataset root path
        if dataset_name == "vitra":
            dataset_root = Path(data_root_dir) / vitra_subdir
            print(f"VITRA dataset root: {dataset_root}")
        elif dataset_name == "real_robot_old":
            dataset_root = Path(data_root_dir) / "yuming" / "Muse_Robot_Dataset"
        elif dataset_name == "real_robot":
            dataset_root = Path(data_root_dir) / "yuming" / "Muse_Robot_Dataset_01"
        elif dataset_name == "molmoact_household":
            dataset_root = Path(data_root_dir) / molmoact_subdir / "molmoact_dataset_household"
        elif dataset_name == "molmoact_tabletop":
            dataset_root = Path(data_root_dir) / molmoact_subdir / "molmoact_dataset_tabletop"
        elif dataset_name == "interna1":
            dataset_root = Path(data_root_dir) / "InternA1_Robot_Data" / "real" / "genie1"
        elif dataset_name == "agibot":
            dataset_root = Path(data_root_dir) / "Agibot_Data_Copy"
        elif dataset_name == "real_robot_merged":
            dataset_root = Path(data_root_dir) / "yuming" / "Muse_Robot_Dataset_01"
            extra_roots = [str(Path(data_root_dir) / "yuming" / "Muse_Robot_Dataset_02")]
        elif dataset_name == "real_robot_merged_test":
            dataset_root = Path(data_root_dir) / "test_dataset_01"
            extra_roots = [str(Path(data_root_dir) / "test_dataset_02")]
        elif dataset_name == "muse_robot":
            # MuseVLA released dataset (HuggingFace): two parts under data_root_dir
            dataset_root = Path(data_root_dir) / "dataset_01"
            extra_roots = [str(Path(data_root_dir) / "dataset_02")]
        elif dataset_name == "hand_pretrain":
            dataset_root = Path(data_root_dir) / "Hand_Pretrain_ReSeg"
        else:
            print(f"Warning: Unknown dataset name: {dataset_name}, skipping...")
            continue

        if not dataset_root.exists():
            print(f"Warning: Dataset root not found: {dataset_root}, skipping...")
            continue

        # Determine statistics path for this dataset
        if dataset_name == "vitra":
            # VITRA and real_robot use the same robot hardware,
            # so share the same normalization statistics for consistent inference
            stats_file = Path(statistics_path) / "yuming" / "Muse_Robot_Dataset_01" / "statistics.json"
        elif dataset_name == "real_robot":
            stats_file = Path(statistics_path) / "yuming" / "Muse_Robot_Dataset_01" / "statistics.json"
        elif dataset_name == "real_robot_old":
            # Build mode suffix matching the convention
            old_mode_suffix = ''
            if vlm_vla_config.get('use_end_pose', False):
                if vlm_vla_config.get('use_delta_pose', False):
                    old_mode_suffix += '_6dpose_delta'
                else:
                    old_mode_suffix += '_6dpose'
            if not vlm_vla_config.get('relative_to_t0', False):
                old_mode_suffix += '_NRt0'
            stats_file = Path(statistics_path) / "yuming" / "Muse_Robot_Dataset" / f"statistics{old_mode_suffix}.json"
        elif "molmoact" in dataset_name:
            split = dataset_name.split('_')[1]
            # Build mode-specific filename matching RoboDatasetVLMVLACore convention
            stats_suffix = ''
            if vlm_vla_config.get('use_end_pose', False):
                if vlm_vla_config.get('use_delta_pose', False):
                    stats_suffix += '_6dpose_delta'
                else:
                    stats_suffix += '_6dpose'
            if not vlm_vla_config.get('relative_to_t0', False):
                stats_suffix += '_NRt0'
            if stats_suffix:
                stats_name = f"statistics{stats_suffix}.json"
            else:
                stats_name = "statistics.json"
            stats_file = Path(statistics_path) / molmoact_subdir / f"molmoact_dataset_{split}" / stats_name
            # Fallback to base statistics.json if mode-specific not found
            if not stats_file.exists():
                stats_file = Path(statistics_path) / molmoact_subdir / f"molmoact_dataset_{split}" / "statistics.json"
        elif dataset_name == "interna1":
            stats_file = Path(statistics_path) / "InternA1_Robot_Data" / "real" / "genie1" / "statistics.json"
        elif dataset_name == "agibot":
            stats_file = Path(statistics_path) / "Agibot_Data_Copy" / "statistics.json"
        elif dataset_name in ("real_robot_merged", "real_robot_merged_test"):
            # Build mode suffix matching the convention in calculate_statistics.py
            merged_mode_suffix = ''
            if vlm_vla_config.get('use_end_pose', False):
                if vlm_vla_config.get('use_delta_pose', False):
                    merged_mode_suffix += '_6dpose_delta'
                else:
                    merged_mode_suffix += '_6dpose'
            if not vlm_vla_config.get('relative_to_t0', False):
                merged_mode_suffix += '_NRt0'

            if dataset_name == "real_robot_merged":
                # Use existing statistics file (same as real_robot, without _merged suffix)
                stats_file = Path(statistics_path) / "yuming" / "Muse_Robot_Dataset_01" / f"statistics{merged_mode_suffix}.json"
            else:  # real_robot_merged_test
                stats_file = Path(statistics_path) / "test_dataset_01" / f"statistics_merged{merged_mode_suffix}.json"
        elif dataset_name == "muse_robot":
            # Build mode suffix and load shared statistics from dataset_01
            muse_mode_suffix = ''
            if vlm_vla_config.get('use_end_pose', False):
                if vlm_vla_config.get('use_delta_pose', False):
                    muse_mode_suffix += '_6dpose_delta'
                else:
                    muse_mode_suffix += '_6dpose'
            if not vlm_vla_config.get('relative_to_t0', False):
                muse_mode_suffix += '_NRt0'
            stats_file = Path(statistics_path) / "dataset_01" / f"statistics{muse_mode_suffix}.json"
        elif dataset_name == "hand_pretrain":
            stats_suffix = ''
            if vlm_vla_config.get('use_end_pose', False):
                if vlm_vla_config.get('use_delta_pose', False):
                    stats_suffix += '_6dpose_delta'
                else:
                    stats_suffix += '_6dpose'
            if not vlm_vla_config.get('relative_to_t0', False):
                stats_suffix += '_NRt0'
            if stats_suffix:
                stats_name = f"statistics{stats_suffix}.json"
            else:
                stats_name = "statistics.json"
            stats_file = Path(statistics_path) / "Hand_Pretrain_ReSeg" / stats_name
            if not stats_file.exists():
                stats_file = Path(statistics_path) / "Hand_Pretrain_ReSeg" / "statistics.json"
        else:
            stats_file = Path(statistics_path) / "statistics.json"  # Fallback
        
        if not stats_file.exists(): 
            if dataset_name == "agibot":
                # Build mode suffix matching the convention in RoboDatasetVLMVLACore
                agibot_mode_suffix = ''
                if vlm_vla_config.get('use_end_pose', False):
                    if vlm_vla_config.get('use_delta_pose', False):
                        agibot_mode_suffix += '_6dpose_delta'
                    else:
                        agibot_mode_suffix += '_6dpose'
                if not vlm_vla_config.get('relative_to_t0', False):
                    agibot_mode_suffix += '_NRt0'
                stats_dir = Path(statistics_path) / "Agibot_Data_Copy"
                agihand_file = stats_dir / f"statistics_agihand{agibot_mode_suffix}.json"
                gripper_file = stats_dir / f"statistics_gripper{agibot_mode_suffix}.json"
                if not agihand_file.exists() or not gripper_file.exists():
                    print(f"Warning: Agibot statistics files not found: {agihand_file}, {gripper_file}, skipping dataset {dataset_name}")
                    continue
            else:
                print(f"Warning: Statistics file not found: {stats_file}, skipping dataset {dataset_name}")
                continue
            
        
        # Create VLM-VLA dataset
        vlm_vla_dataset = RoboDatasetVLMVLACore(
            dataset_name=dataset_name,
            root_dir=str(dataset_root),
            statistics_path=str(stats_file),
            sensor_token_prob=vlm_vla_config.get('sensor_token_prob', {
                'none': 0.25, 'thermal': 0.25, 'radar': 0.25, 'audio': 0.25
            }),
            background_modes=vlm_vla_config.get('background_modes', "grey"),
            placeholder_replacements_path=vlm_vla_config.get('placeholder_replacements_path', None),
            placeholder_text=vlm_vla_config.get('placeholder_text', '<placeholder>'),
            processor=processor,
            action_past_window_size=0,
            action_future_window_size=action_future_window_size,
            image_past_window_size=0,
            image_future_window_size=0,
            seed=seed + shard_index,
            max_samples=vlm_vla_config.get('max_samples', None),
            action_type=action_type,
            random_sample_sensor=vlm_vla_config.get('random_sample_sensor', True),
            sensor_masked=vlm_vla_config.get('sensor_masked', True),
            data_mix_sensor=vlm_vla_config.get('data_mix_sensor', "acoustic, radar, thermal"),
            jump_frame=vlm_vla_config.get('jump_frame', 4),
            adaptive_jump_frame=vlm_vla_config.get('adaptive_jump_frame', 2),
            threshold=vlm_vla_config.get('threshold', 0.087),
            sensor_input_all=vlm_vla_config.get('sensor_input_all', False),
            RGB_only=vlm_vla_config.get('RGB_only', False),
            use_end_pose=vlm_vla_config.get('use_end_pose', False),
            use_delta_pose=vlm_vla_config.get('use_delta_pose', False),
            relative_to_t0=vlm_vla_config.get('relative_to_t0', False),
            random_drop_RGB=vlm_vla_config.get('random_drop_RGB', 0),
            radar_fix_start_frame=vlm_vla_config.get('radar_fix_start_frame', False),
            use_non_vitra_ee_action=vlm_vla_config.get('use_non_vitra_ee_action', True),
            data_fraction=vlm_vla_config.get('vitra_data_fraction', vlm_vla_config.get('data_fraction', 1.0)) if dataset_name == 'vitra' else vlm_vla_config.get('data_fraction', 1.0),
            filter_invalid_videos=vlm_vla_config.get('filter_invalid_videos', True),
            train_seg_token=vlm_vla_config.get('train_seg_token', False),
            text_generalization_prob=vlm_vla_config.get('text_generalization_prob', 0.0),
            visual_clone_prob=vlm_vla_config.get('visual_clone_prob', 0.0),
            extra_roots=extra_roots,
            use_validity_filter=vlm_vla_config.get('use_validity_filter', True),
        )
        
        datasets.append(vlm_vla_dataset)
    
    # Create weighted dataset
    if len(datasets) == 0:
        raise ValueError("No valid datasets found!")
    
    # Filter weights to match actual datasets loaded
    final_weights = [dataset_weights[i] for i, name in enumerate(dataset_names) 
                     if any(d.dataset_name == name for d in datasets)]
    
    combined_dataset = MultipleWeightedDataset(datasets, weights=final_weights)
    
    # Create batch sampler for weighted sampling
    batch_sampler = MultipleDatasetWeightedDistributedBatchSampler(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        seed=seed,
        num_replicas=shard_num,
        rank=shard_index,
    )
    
    # Create collator for padding
    collator = PaddedCollatorForVLMAndVLA(
        model_max_length=processor.tokenizer.model_max_length,
        pad_token_id=processor.tokenizer.pad_token_id,
        padding_side=padding_side,
    )
    
    return combined_dataset, collator, batch_sampler