# Sensor Calibration

Tools for calibrating and aligning multimodal sensors with the RGB camera.

## Overview

MuseVLA uses grounded sensor images that overlay sensor heatmaps on RGB images. This requires computing homography matrices to transform each sensor's coordinate frame to the RGB camera frame.

## Supported Sensors

| Sensor | Model | Calibration Method |
|--------|-------|-------------------|
| Thermal | infiRay T2S+ | Checkerboard homography |
| Acoustic | Sipeed 6+1 Mic Array | Sound source localization |
| Radar | Calterah 4T4R 60GHz | Corner reflector alignment |

## Directory Structure

```
calibration/
├── src/                    # Calibration source code
│   ├── thermal/           # Thermal camera calibration
│   ├── acoustic/          # Acoustic array calibration
│   └── radar/             # mmWave radar calibration
├── homography/            # Pre-computed homography matrices
│   ├── thermal/{left,right}/homography_rgb_to_ir.npy
│   ├── acoustic/homography_rgb_to_acoustic.npy
│   └── radar/homography_rgb_to_radar.npy
└── scripts/               # Calibration scripts
```

## Pre-computed Matrices

The `homography/` directory contains pre-computed homography matrices for our sensor setup. These can be used directly if your sensor placement is similar.

## Custom Calibration

To calibrate for your own sensor setup:

```bash
# Thermal camera
python scripts/thermal/homography_calculator.py \
    --rgb-image <RGB_CALIBRATION_IMAGE> \
    --thermal-image <THERMAL_CALIBRATION_IMAGE>

# Acoustic array
python scripts/acoustic/homography_calculator.py \
    --rgb-image <RGB_CALIBRATION_IMAGE> \
    --acoustic-data <ACOUSTIC_RECORDING>

# Radar
python scripts/radar/homography_calculator.py \
    --rgb-image <RGB_CALIBRATION_IMAGE> \
    --radar-data <RADAR_RECORDING>
```

## Usage in Inference

The inference module automatically loads homography matrices from `calibration/homography/` to transform sensor data to RGB coordinates before overlay.
