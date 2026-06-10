# Inference Module

This module provides inference classes for deploying MuseVLA models on real robots.

## Overview

MuseVLA uses a three-stage inference pipeline:

1. **Sensor Selection**: VLM predicts which sensor modality to use (`<thermal>`, `<radar>`, `<audio>`, or `<None>`)
2. **Object Segmentation**: SAM3 segments the target object and overlays sensor data
3. **Action Prediction**: VLA generates action sequences from the grounded sensor image

## Inference Classes

| Class | State Dim | Action Dim | Description |
|-------|-----------|------------|-------------|
| `MuseVLAInference6DPos` | 18 | 18 | 6D EEF pose + 12D hand joints |
| `MuseVLAInference6DPosDelta` | 18 | 18 | 6D EEF pose delta + 12D hand joints |
| `MuseVLAInference7DJointAngle` | 19 | 19 | 7D arm joints + 12D hand joints |

## Usage

```python
from inference import MuseVLAInference6DPosDelta

# Initialize
inferencer = MuseVLAInference6DPosDelta(
    vla_config_path="musevla/configs/robot_vlm_vla_finetune_dist.json",
    pretrained_path="<CHECKPOINT_PATH>",
    statistics_path="<STATISTICS_PATH>",
    sam3_checkpoint_path="<SAM3_PATH>",
    calibration_dir="calibration",
)

# Run inference
action = inferencer.inference(
    rgb_image=rgb_image,           # (H, W, 3) numpy array
    instruction="Pick up the hot cup",
    state=current_state,           # (18,) numpy array
    sensor_images={
        "thermal_left": thermal_img,
        "acoustic": acoustic_img,
        "radar": radar_img,
    },
)
# Returns: (chunk_size, 18) action sequence
```

## Required Files

- Model checkpoint (`.ckpt`)
- Dataset statistics (`statistics.json`)
- SAM3 checkpoint
- Sensor calibration matrices (in `calibration/homography/`)

## Extending

To support new action formats, subclass `MuseVLAInferenceBase`:

```python
from inference.base import MuseVLAInferenceBase

class CustomInference(MuseVLAInferenceBase):
    STATE_DIM = 20
    ACTION_DIM = 20
    # Implement required methods...
```
