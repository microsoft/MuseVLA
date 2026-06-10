"""
MuseVLA Inference API Server - 6D Position Delta Output

A Flask-based API server for MuseVLA model inference.
Receives robot state and sensor images, returns predicted actions.

Output format: 18D per hand (6D EEF pose delta + 12D hand joints)

Usage:
    # Use YAML config file (recommended)
    python scripts/inference_api_server_6d_pos_delta.py \
        --config data/robot_check/configs/xxx.yaml

    # Use CLI args (legacy)
    python scripts/inference_api_server_6d_pos_delta.py \
        --vla_config musevla/configs/vlm_vla_cofinetune_single.json \
        --pretrained_path /path/to/checkpoint \
        --statistics_path /path/to/statistics.json \
        --sam3_checkpoint /path/to/sam3.pt \
        --port 7500
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from flask import Flask, jsonify, request
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inference import MuseVLAInference6DPosDelta


app = Flask(__name__)

# Global inference engine (initialized on startup)
inferencer: MuseVLAInference6DPosDelta = None

USE_RULE_BASED = False  # Set to True to enable rule-based

def decode_image_file(file) -> np.ndarray:
    """Decode uploaded image file to numpy array."""
    np_arr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

@app.route('/api/inference', methods=['POST'])
def inference_endpoint():
    """
    Inference API endpoint.

    Request:
        - images: Main camera image (required)
        - json: JSON configuration (required)
            {
                "task_description": str,
                "state": list[float],  # 18D
                "cfg_scale": float (default: 5.0),
                "num_ddim_steps": int (default: 10),
            }
        - acoustic_image: Acoustic image (optional)
        - thermal_left_image: Left thermal image (optional)
        - thermal_right_image: Right thermal image (optional)
        - radar_image: Radar image (optional)

    Response:
        list[list[float]]  # [chunk_size, 18]
    """
    global inferencer

    if inferencer is None:
        return jsonify({"error": "Inference engine not initialized"}), 500

    try:
        if 'images' not in request.files:
            return jsonify({"error": "Missing 'images' file"}), 400

        rgb_image = decode_image_file(request.files['images'])

        if 'json' not in request.files:
            return jsonify({"error": "Missing 'json' file"}), 400

        config = json.load(request.files['json'])

        task_description = config.get('task_description', '')
        state = config.get('state', [])
        cfg_scale = config.get('cfg_scale', 5.0)
        num_ddim_steps = config.get('num_ddim_steps', 10)

        if len(state) != 18:
            return jsonify({
                "error": f"State must be 18D, got {len(state)}D"
            }), 400

        state_18d = np.array(state, dtype=np.float32)

        # Parse optional sensor images
        sensor_images = {}

        if 'acoustic_image' in request.files:
            sensor_images['acoustic'] = decode_image_file(request.files['acoustic_image'])

        if 'thermal_left_image' in request.files:
            sensor_images['thermal_left'] = decode_image_file(request.files['thermal_left_image'])

        if 'thermal_right_image' in request.files:
            sensor_images['thermal_right'] = decode_image_file(request.files['thermal_right_image'])

        if 'radar_image' in request.files:
            sensor_images['radar'] = decode_image_file(request.files['radar_image'])

        # Run inference
        action_18d = inferencer.inference(
            rgb_image=rgb_image,
            instruction=task_description,
            state=state_18d,
            sensor_images=sensor_images,
            num_ddim_steps=num_ddim_steps,
            cfg_scale=cfg_scale,
            use_rule_based=USE_RULE_BASED,
        )

        action_list = action_18d.tolist()
        print(f"Returning action with shape: {action_18d.shape}")

        return jsonify(action_list)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    global inferencer
    return jsonify({
        "status": "ok" if inferencer is not None else "not_initialized",
        "chunk_size": inferencer.chunk_size if inferencer else None,
        "output_format": "6d_pos_delta",
        "state_dim": 18,
        "action_dim": 18,
    })


@app.route('/api/clear_cache', methods=['POST'])
def clear_cache():
    """Clear VLM cache endpoint."""
    global inferencer
    if inferencer:
        inferencer.clear_cache()
        return jsonify({"status": "cache_cleared"})
    return jsonify({"status": "no_cache"})


def load_config_from_yaml(yaml_path: str) -> dict:
    """Load inference-related config from YAML file.

    Only extracts fields relevant to inference server, ignoring
    visualization/rendering/dataset settings.
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Extract only inference-related fields
    config = {}
    field_mapping = {
        'pretrained_path': 'pretrained_path',
        'statistics_path': 'statistics_path',
        'vla_config': 'vla_config',
        'sam3_checkpoint': 'sam3_checkpoint',
        'calibration_dir': 'calibration_dir',
    }

    for yaml_key, config_key in field_mapping.items():
        if yaml_key in data and data[yaml_key] is not None:
            config[config_key] = str(data[yaml_key])

    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="MuseVLA Inference API Server (6D Position Delta Output)"
    )
    parser.add_argument(
        "--config", "-c", type=str, required=False,
        help="Path to YAML config file (fields can be overridden by CLI args)"
    )
    parser.add_argument(
        "--vla_config", type=str, required=False,
        default=None,
        help="Path to VLA config JSON"
    )
    parser.add_argument(
        "--pretrained_path", type=str, required=False,
        default=None,
        help="Path to pretrained checkpoint"
    )
    parser.add_argument(
        "--statistics_path", type=str, required=False,
        default=None,
        help="Path to statistics.json for normalization"
    )
    parser.add_argument(
        "--sam3_checkpoint", type=str, required=False,
        default=None,
        help="Path to SAM3 checkpoint"
    )
    parser.add_argument(
        "--calibration_dir", type=str, default=None,
        help="Path to calibration directory"
    )
    parser.add_argument(
        "--port", type=int, default=7500,
        help="Server port (default: 7500)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--no_vlm_cache", action="store_true",
        help="Disable VLM caching"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Default values (used when neither config nor CLI provides a value)
    defaults = {
        'vla_config': 'musevla/configs/robot_vlm_vla_finetune_dist.json',
        'pretrained_path': None,
        'statistics_path': None,
        'sam3_checkpoint': 'assets/sam3.pt',
        'calibration_dir': 'calibration',
    }

    # Load config from YAML if provided
    config_values = {}
    if args.config:
        print(f"Loading config from: {args.config}")
        config_values = load_config_from_yaml(args.config)

    # Merge: CLI args > config file > defaults
    def get_value(key):
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            return cli_val
        if key in config_values:
            return config_values[key]
        return defaults.get(key)

    vla_config = get_value('vla_config')
    pretrained_path = get_value('pretrained_path')
    statistics_path = get_value('statistics_path')
    sam3_checkpoint = get_value('sam3_checkpoint')
    calibration_dir = get_value('calibration_dir')

    # Validate required fields
    if not pretrained_path:
        raise ValueError("pretrained_path is required (via --config or --pretrained_path)")
    if not statistics_path:
        raise ValueError("statistics_path is required (via --config or --statistics_path)")

    print(f"Config values:")
    print(f"  vla_config: {vla_config}")
    print(f"  pretrained_path: {pretrained_path}")
    print(f"  statistics_path: {statistics_path}")
    print(f"  sam3_checkpoint: {sam3_checkpoint}")
    print(f"  calibration_dir: {calibration_dir}")

    # Initialize inference engine
    inferencer = MuseVLAInference6DPosDelta(
        vla_config_path=vla_config,
        pretrained_path=pretrained_path,
        statistics_path=statistics_path,
        sam3_checkpoint_path=sam3_checkpoint,
        calibration_dir=calibration_dir,
        use_vlm_cache=not args.no_vlm_cache,
    )

    print(f"\nStarting server on {args.host}:{args.port}")
    print(f"VLM cache: {'disabled' if args.no_vlm_cache else 'enabled'}")
    print(f"Output format: 6D Position Delta (18D per hand)")

    app.run(host=args.host, port=args.port, debug=False)
