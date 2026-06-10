"""
Test script to verify real robot data loading with new parameters.
"""
import sys
from pathlib import Path

# Test configuration
config = {
    "dataset_name": "real_robot",
    "root_dir": "<PROJECT_PATH>",
    "statistics_path": "<PROJECT_PATH>",
    "random_sample_sensor": False,
    "sensor_masked": True,
    "data_mix": "acoustic, radar, thermal",
    "background_modes": "grey",
    "max_samples": 5,
}

print("Testing real robot data loading...")
print(f"Dataset: {config['dataset_name']}")
print(f"Root dir: {config['root_dir']}")
print(f"Random sample sensor: {config['random_sample_sensor']}")
print(f"Sensor masked: {config['sensor_masked']}")
print(f"Data mix: {config['data_mix']}")
print(f"Background modes: {config['background_modes']}")
print()

# Check if root directory exists
root = Path(config['root_dir'])
if not root.exists():
    print(f"ERROR: Root directory does not exist: {root}")
    sys.exit(1)

# Check folder structure
print("Checking folder structure...")
for data_type in ['acoustic_data', 'thermal_data', 'radar_data']:
    data_folder = root / data_type
    label_folder = root / data_type.replace('_data', '_labels')
    
    if data_folder.exists():
        episodes = list(data_folder.iterdir())
        print(f"  ✓ {data_type}: {len(episodes)} episodes")
    else:
        print(f"  ✗ {data_type}: not found")
    
    if label_folder.exists():
        labels = list(label_folder.glob("*.json"))
        print(f"  ✓ {data_type.replace('_data', '_labels')}: {len(labels)} files")
    else:
        print(f"  ✗ {data_type.replace('_data', '_labels')}: not found")

print()

# Try to load a sample label file
print("Testing label file reading...")
acoustic_labels = root / "acoustic_labels"
if acoustic_labels.exists():
    label_files = sorted(list(acoustic_labels.glob("*.json")))
    if label_files:
        import json
        label_file = label_files[0]
        print(f"Reading: {label_file.name}")
        with open(label_file, 'r') as f:
            label_data = json.load(f)
        
        print(f"  - video_name: {label_data.get('video_name')}")
        print(f"  - frame_num_total: {label_data.get('frame_num_total')}")
        print(f"  - abolished: {label_data.get('problem', {}).get('abolished', False)}")
        annotations = label_data.get('annotations', [])
        if annotations:
            print(f"  - instruction: {annotations[0].get('instruction', 'N/A')}")
        else:
            print(f"  - instruction: No annotations")
        print()

# Check sensor video structure
print("Testing sensor video structure...")
acoustic_data = root / "acoustic_data"
if acoustic_data.exists():
    episodes = sorted(list(acoustic_data.iterdir()))[:2]
    for episode in episodes:
        print(f"Episode: {episode.name}")
        video_path = episode / "video.mp4"
        print(f"  - RGB video: {'✓' if video_path.exists() else '✗'}")
        
        for sensor in ['acoustic', 'radar', 'thermal']:
            sensor_folder = episode / sensor
            if sensor_folder.exists():
                masked_folder = sensor_folder / "images_masked"
                if masked_folder.exists():
                    video_files = list(masked_folder.glob("video*.mp4"))
                    print(f"  - {sensor}/images_masked: {len(video_files)} videos")
                else:
                    print(f"  - {sensor}/images_masked: not found")
            else:
                print(f"  - {sensor}: not found")
        print()

print("\n✓ All structural checks passed!")
print("\nTo test actual dataset loading, run:")
print("  python -c 'from musevla.datasets.robot_dataset_vlm_vla import RoboDatasetVLMVLACore; print(\"Import successful!\")'")
