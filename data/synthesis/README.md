# Data Synthesis

This module generates multisensory training data from existing RGB robot datasets by injecting physical property keywords and overlaying synthetic sensor masks.

## Overview

The data synthesis pipeline enables training MuseVLA without collecting real sensor data by:

1. **Keyword Injection**: Adding physical property descriptors (e.g., "hot", "cold", "ringing") to task instructions
2. **Object Segmentation**: Using SAM3 to segment target objects in RGB frames
3. **Sensor Overlay**: Applying color-coded sensor masks on segmented regions

## Scripts

| Script | Description |
|--------|-------------|
| `data_synthesis_vitra.py` | Synthesis for VITRA dataset |
| `data_synthesis_molmoact.py` | Synthesis for MolmoAct dataset |
| `data_synthesis_agibot.py` | Synthesis for AgiBotWorld dataset |
| `*_dist.py` | Distributed versions (multi-GPU) |
| `*_reseg.py` | Re-segmentation for failed samples |

## Core Utilities

| File | Description |
|------|-------------|
| `sensor_mask_utils.py` | Sensor overlay generation |
| `sensor_mask_token_utils.py` | Mask tokenization for PaliGemma |
| `vlm_utils.py` | VLM inference utilities |

## Usage

### Single Process

```bash
python data_synthesis_vitra.py \
    --data-root <DATA_ROOT>/VITRA_Robot_Data \
    --output-root <OUTPUT_ROOT>/VITRA_Robot_Data_Syn
```

### Distributed (Multi-GPU)

```bash
torchrun --nproc_per_node=8 data_synthesis_vitra_dist.py \
    --data-root <DATA_ROOT>/VITRA_Robot_Data \
    --output-root <OUTPUT_ROOT>/VITRA_Robot_Data_Syn
```

## Dependencies

- SAM3 for segmentation (see Installation in main README)
- PyTorch with CUDA support
- OpenCV, NumPy, PIL

## Output Format

Each synthesized episode contains:
- `video.mp4` - Original RGB video
- `video_seg.mp4` - Segmentation mask video
- `metadata.json` - Episode metadata with sensor type and keywords
