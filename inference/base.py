"""
MuseVLA Inference Base Class

Provides the core inference engine for MuseVLA models with:
- VLM prediction (Stage 1)
- SAM3 segmentation + Sensor overlay (Stage 2)
- VLA action prediction (Stage 3)

Subclasses define the state/action dimensions for different output formats.
"""

import json
import os
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "musevla" / "datasets"))
sys.path.insert(0, str(project_root / "calibration"))
sys.path.insert(0, str(project_root / "third_party" / "sam3"))

from acoustic_overlay_optimized import overlay_acoustic_on_image_optimized_bgr
from radar_overlay_optimized import overlay_radar_on_image_optimized_bgr
from thermal_overlay_optimized import overlay_thermal_on_image_optimized_bgr

from src.thermal import ThermalMaskGenerator
from src.radar import RadarMaskGenerator
from src.acoustic import AcousticMaskGenerator

from musevla.models import load_model
from musevla.utils.data_utils import load_normalizer
from musevla.datasets.dataset_utils import ActionFeature, StateFeature
from musevla.datasets.robot_dataset import (
    transfer_xhand_to_human,
    transfer_human_to_xhand,
    pad_state_robot,
    pad_action,
)

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


SENSOR_TOKENS = ["None", "thermal", "radar", "audio"]
OBJECT_TOKEN = ["bottle", "bottles", "box", "boxes", "drink", "towel", "towels", "clothes"]


class VLMCache:
    """Cache for VLM prediction results to avoid redundant computation."""

    def __init__(self):
        self.last_task_description = None
        self.cached_result = None

    def get_or_compute(self, task_description: str, compute_fn):
        """Get cached result if task_description matches, otherwise compute and cache."""
        if task_description == self.last_task_description and self.cached_result is not None:
            print(f"  [VLM Cache] Using cached result for: {task_description[:50]}...")
            return self.cached_result

        print(f"  [VLM Cache] Computing new result for: {task_description[:50]}...")
        result = compute_fn()
        self.last_task_description = task_description
        self.cached_result = result
        return result

    def clear(self):
        """Clear the cache."""
        self.last_task_description = None
        self.cached_result = None


def generate_ground_truth_sensor_and_object(instruction: str) -> tuple:
    """Extract sensor type and object description from instruction text."""
    sensor = "none"
    object_description = "object"

    for object_token in OBJECT_TOKEN:
        if object_token in instruction:
            object_description = object_token
            if object_description == "drink":
                object_description = "bottle"
            break

    print("Generating ground truth sensor and object from instruction:", instruction)

    if instruction == "Pick up the item in the box.":
        sensor = "none"
    elif "drink in the box" in instruction:
        sensor = "thermal"
        object_description = "small box"
    elif "hot tomato" in instruction:
        sensor = "thermal"
        object_description = "red item"
    elif "occupied bag" in instruction:
        sensor = "radar"
        object_description = "white bag"
    elif "bag covering" in instruction:
        sensor = "audio"
        object_description = "white bag"
    elif "ringing" in instruction or "sound" in instruction or "audio" in instruction:
        sensor = "audio"
    elif "occupied" in instruction or "obstacle" in instruction or "radar" in instruction:
        sensor = "radar"
    elif "hot" in instruction or "temperature" in instruction or "cold" in instruction:
        sensor = "thermal"

    return sensor, object_description


class MuseVLAInferenceBase(ABC):
    """
    Base class for MuseVLA three-stage inference engine.

    Subclasses must define:
    - STATE_DIM: Input state dimension per hand
    - ACTION_DIM: Output action dimension per hand
    - FULL_STATE_DIM: Full state dimension (both hands)
    - FULL_ACTION_DIM: Full action dimension (both hands)
    """

    # Subclasses must define these
    STATE_DIM: int  # Per hand (e.g., 18 for 6D, 19 for 7D)
    ACTION_DIM: int  # Per hand
    FULL_STATE_DIM: int  # Both hands (e.g., 36 or 38)
    FULL_ACTION_DIM: int  # Both hands

    def __init__(
        self,
        vla_config_path: str,
        pretrained_path: str,
        statistics_path: str,
        sam3_checkpoint_path: str,
        calibration_dir: str,
        use_vlm_cache: bool = False,
        sam3_confidence_threshold: float = 0.7,
    ):
        """
        Initialize the MuseVLA inference engine.

        Args:
            vla_config_path: Path to VLA config JSON
            pretrained_path: Path to pretrained checkpoint
            statistics_path: Path to statistics.json for normalization
            sam3_checkpoint_path: Path to SAM3 checkpoint
            calibration_dir: Path to calibration directory
            use_vlm_cache: Whether to cache VLM results
            sam3_confidence_threshold: Confidence threshold for SAM3 segmentation (0.0-1.0)
        """
        print("=" * 80)
        print(f"Initializing {self.__class__.__name__}")
        print("=" * 80)

        # Load VLA config
        print(f"Loading VLA config from: {vla_config_path}")
        with open(vla_config_path, 'r') as f:
            self.configs = json.load(f)

        self.configs['model_load_path'] = pretrained_path
        self.configs['statistics_path'] = statistics_path

        # Load VLA model
        print(f"Loading VLA model from: {pretrained_path}")
        t0 = time.time()
        self.model = load_model(self.configs).cuda()
        self.model.eval()
        print(f"  VLA model loaded in {time.time() - t0:.1f}s")

        # Load normalizer
        print(f"Loading normalizer from: {statistics_path}")
        self.normalizer = load_normalizer(self.configs)

        # Load SAM3 model
        print(f"Loading SAM3 model from: {sam3_checkpoint_path}")
        t0 = time.time()
        self.sam3_model = build_sam3_image_model(checkpoint_path=sam3_checkpoint_path)
        self.sam3_processor = Sam3Processor(
            self.sam3_model, confidence_threshold=sam3_confidence_threshold
        )
        self.sam3_confidence_threshold = sam3_confidence_threshold
        print(f"  SAM3 model loaded in {time.time() - t0:.1f}s")
        print(f"  SAM3 confidence threshold: {sam3_confidence_threshold}")

        # Load calibration data
        self.calibration_dir = Path(calibration_dir)
        self._load_calibration_data()

        # Initialize VLM cache
        self.use_vlm_cache = use_vlm_cache
        self.vlm_cache = VLMCache() if use_vlm_cache else None

        # Get chunk size from model
        self.chunk_size = self.model.chunk_size

        print("=" * 80)
        print(f"{self.__class__.__name__} initialized successfully")
        print(f"  Chunk size: {self.chunk_size}")
        print(f"  State dim: {self.STATE_DIM} (per hand), {self.FULL_STATE_DIM} (total)")
        print(f"  Action dim: {self.ACTION_DIM} (per hand), {self.FULL_ACTION_DIM} (total)")
        print("=" * 80)

    def _load_calibration_data(self):
        """Load homography matrices for sensor overlay."""
        print("Loading calibration data...")

        # Thermal calibration
        self.thermal_left_H = np.load(
            self.calibration_dir / "homography/thermal/left/homography_rgb_to_ir.npy"
        )
        self.thermal_right_H = np.load(
            self.calibration_dir / "homography/thermal/right/homography_rgb_to_ir.npy"
        )
        self.thermal_generator = ThermalMaskGenerator(
            rgb_to_left_thermal_transform=self.thermal_left_H,
            rgb_to_right_thermal_transform=self.thermal_right_H,
        )

        # Radar calibration
        self.radar_H = np.load(
            self.calibration_dir / "homography/radar/homography_rgb_to_radar.npy"
        )
        self.radar_generator = RadarMaskGenerator(rgb_to_radar_transform=self.radar_H)

        # Acoustic calibration
        self.acoustic_H = np.load(
            self.calibration_dir / "homography/acoustic/homography_rgb_to_acoustic.npy"
        )
        self.acoustic_generator = AcousticMaskGenerator(
            rgb_to_acoustic_transform=self.acoustic_H
        )

        print("  Calibration data loaded successfully")

    # =========================================================================
    # Abstract methods for dimension conversion (subclass must implement)
    # =========================================================================

    @abstractmethod
    def convert_input_state(self, state: np.ndarray) -> np.ndarray:
        """
        Convert client's per-hand state to full state.

        Args:
            state: Input state array (STATE_DIM,)

        Returns:
            Full state array (FULL_STATE_DIM,)
        """
        pass

    @abstractmethod
    def convert_output_action(self, action: np.ndarray) -> np.ndarray:
        """
        Convert model's full action to per-hand action.

        Args:
            action: Full action array (chunk_size, FULL_ACTION_DIM)

        Returns:
            Per-hand action array (chunk_size, ACTION_DIM)
        """
        pass

    @abstractmethod
    def prepare_state_and_masks(
        self,
        state_full: np.ndarray,
        state_mask: Optional[np.ndarray],
        action_mask: Optional[np.ndarray],
    ) -> tuple:
        """
        Prepare state tensor and masks for VLA prediction.

        Args:
            state_full: Full state array (FULL_STATE_DIM,)
            state_mask: Optional state mask
            action_mask: Optional action mask

        Returns:
            Tuple of (unified_state, unified_state_mask, unified_action_mask, extra_data)
        """
        pass

    @abstractmethod
    def postprocess_action(
        self,
        norm_action: np.ndarray,
        extra_data: dict,
    ) -> np.ndarray:
        """
        Post-process normalized action from model.

        Args:
            norm_action: Normalized action from model
            extra_data: Extra data from prepare_state_and_masks

        Returns:
            Unnormalized action array (chunk_size, FULL_ACTION_DIM)
        """
        pass

    # =========================================================================
    # Stage 1: VLM Prediction
    # =========================================================================

    def predict_vlm(
        self,
        rgb_image: np.ndarray,
        instruction: str,
        use_rule_based: bool = False,
        debug: bool = False,
    ) -> dict:
        """
        Stage 1: VLM prediction for object detection and sensor selection.

        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            instruction: Task instruction string (raw task description, no format prefix)
            use_rule_based: If True, use rule-based matching instead of VLM output.
                           If False, use VLM prediction with rule-based fallback.
            debug: If True, print debug information

        Returns:
            dict with keys:
                - predicted_sensor: str ('thermal', 'radar', 'audio', 'none')
                # - predicted_text: str (VLM generated text)
                - raw_generated_text: str (raw text generated by VLM before any parsing)
                - object_description: str (extracted object description)
                - source: str ('vlm' or 'rule') indicating which method was used
        """
        # Rule-based matching (used as primary or fallback)
        rule_sensor, rule_object = generate_ground_truth_sensor_and_object(instruction)

        if use_rule_based:
            # Pure rule-based mode: skip VLM inference entirely
            if debug:
                print(f"  [VLM] Using rule-based matching")
                print(f"  [VLM] Rule sensor: {rule_sensor}, object: {rule_object}")
            return {
                'predicted_sensor': rule_sensor,
                'object_description': rule_object,
                'raw_generated_text': '',
                'source': 'rule',
            }

        # VLM-based mode: run VLM inference
        # Note: Do NOT add format prefix here - the model's predict_vlm handles it internally
        # The model expects raw instruction like "pick up the hot bottle"
        # and internally constructs: "<image>Task: {instruction} Question: Choose from..."
        vlm_results = self.model.predict_vlm(
            image=rgb_image,
            instruction=instruction,  # Raw instruction without format prefix
            ground_truth_text=None,
            ground_truth_sensor=None,
            current_state=None,
            current_state_mask=None,
            fov=None,
            max_new_tokens=5,
            temperature=0,
            top_p=1.0,
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
            use_cache=True,
            debug_training=debug,
        )

        object_description = vlm_results.get('predicted_text', '')  # returned predicted text from VLM (object description)
        vlm_sensor = vlm_results.get('predicted_sensor')
        raw_generated_text = vlm_results.get('raw_generated_text', '')

        # Determine final sensor: use VLM output, fallback to rule if VLM fails
        if vlm_sensor is not None and vlm_sensor.lower() in ['thermal', 'radar', 'audio', 'none']:
            predicted_sensor = vlm_sensor.lower()
            source = 'vlm'
        else:
            predicted_sensor = rule_sensor
            source = 'rule'
            if debug:
                print(f"  [VLM] VLM sensor prediction failed, fallback to rule: {rule_sensor}")

        if debug:
            print(f"  [VLM] Source: {source}")
            print(f"  [VLM] Predicted sensor: {predicted_sensor}")
            print(f"  [VLM] Raw generated text: {raw_generated_text}")
            print(f"  [VLM] Object description: {object_description}")

        return {
            'predicted_sensor': predicted_sensor,
            'raw_generated_text': raw_generated_text,
            'object_description': object_description,
            'source': source,
        }

    # =========================================================================
    # Stage 2: SAM3 + Sensor Overlay
    # =========================================================================

    def generate_sensor_masked_image(
        self,
        rgb_image: np.ndarray,
        object_description: str,
        predicted_sensor: str,
        sensor_images: dict,
    ) -> tuple:
        """
        Stage 2: Generate sensor masked image using SAM3.

        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            object_description: Object description from VLM
            predicted_sensor: Predicted sensor type
            sensor_images: dict with sensor images

        Returns:
            Tuple of (sensor_masked_image, combined_mask)
        """
        height, width = rgb_image.shape[:2]

        # Run SAM3 segmentation
        print(f"  Running SAM3 with text prompt: '{object_description}'")
        rgb_pil_image = Image.fromarray(rgb_image)
        inference_state = self.sam3_processor.set_image(rgb_pil_image)
        output = self.sam3_processor.set_text_prompt(
            state=inference_state, prompt=object_description
        )

        # Get masks
        masks = output["masks"]

        # Combine all masks
        if len(masks) > 0:
            masks_np = masks.cpu().numpy() if isinstance(masks, torch.Tensor) else np.array(masks)
            print(f"  Raw masks shape: {masks_np.shape}, dtype: {masks_np.dtype}")

            if masks_np.ndim == 4:
                masks_np = masks_np.squeeze(1)

            if masks_np.ndim == 3:
                combined_mask = np.any(masks_np, axis=0).astype(bool)
            elif masks_np.ndim == 2:
                combined_mask = masks_np.astype(bool)
            else:
                print(f"  Warning: Unexpected mask shape {masks_np.shape}, creating empty mask")
                combined_mask = np.zeros((height, width), dtype=bool)

            if combined_mask.shape != (height, width):
                print(f"  Warning: Mask shape {combined_mask.shape} doesn't match image {(height, width)}")
                if combined_mask.size == height * width:
                    combined_mask = combined_mask.reshape(height, width)
                else:
                    combined_mask = np.zeros((height, width), dtype=bool)

            print(f"  Combined mask shape: {combined_mask.shape}, pixels: {combined_mask.sum()}")
        else:
            combined_mask = np.zeros((height, width), dtype=bool)

        if combined_mask.sum() == 0:
            print("  No valid masks found, using RGB image without overlay")
            return rgb_image.copy(), combined_mask

        print(f"  Generated {len(masks)} masks")
        print(f"  Applying {predicted_sensor} overlay on masked region...")

        # Convert RGB to BGR for optimized functions
        image_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        mask_uint8 = (combined_mask.astype(np.uint8) * 255)

        if predicted_sensor == 'thermal':
            thermal_left = sensor_images.get('thermal_left')
            thermal_right = sensor_images.get('thermal_right')

            if thermal_left is None or thermal_right is None:
                print("  Warning: Missing thermal images, using RGB without overlay")
                return rgb_image.copy(), combined_mask

            thermal_left_bgr = cv2.cvtColor(thermal_left, cv2.COLOR_RGB2BGR)
            thermal_right_bgr = cv2.cvtColor(thermal_right, cv2.COLOR_RGB2BGR)

            dark_result, color_result, grey_result = overlay_thermal_on_image_optimized_bgr(
                thermal_generator=self.thermal_generator,
                bgr_image=image_bgr,
                mask=mask_uint8,
                thermal_left_bgr_image=thermal_left_bgr,
                thermal_right_bgr_image=thermal_right_bgr,
                alpha=1.0,
            )
            sensor_masked_image_bgr = grey_result

        elif predicted_sensor == 'radar':
            radar_image = sensor_images.get('radar')

            if radar_image is None:
                print("  Warning: Missing radar image, using RGB without overlay")
                return rgb_image.copy(), combined_mask

            radar_image_bgr = cv2.cvtColor(radar_image, cv2.COLOR_RGB2BGR)

            dark_result, color_result, grey_result = overlay_radar_on_image_optimized_bgr(
                radar_generator=self.radar_generator,
                bgr_image=image_bgr,
                mask=mask_uint8,
                radar_bgr_image=radar_image_bgr,
                alpha=1.0,
            )
            sensor_masked_image_bgr = grey_result

        elif predicted_sensor == 'audio':
            acoustic_image = sensor_images.get('acoustic')

            if acoustic_image is None:
                print("  Warning: Missing acoustic image, using RGB without overlay")
                return rgb_image.copy(), combined_mask

            acoustic_image_bgr = cv2.cvtColor(acoustic_image, cv2.COLOR_RGB2BGR)

            dark_result, color_result, grey_result = overlay_acoustic_on_image_optimized_bgr(
                acoustic_generator=self.acoustic_generator,
                bgr_image=image_bgr,
                mask=mask_uint8,
                acoustic_bgr_image=acoustic_image_bgr,
                alpha=1.0,
            )
            sensor_masked_image_bgr = grey_result

        else:  # 'none' or unknown sensor
            print(f"  No sensor overlay for sensor type: {predicted_sensor}")
            return np.zeros_like(rgb_image), combined_mask

        sensor_masked_image_np = cv2.cvtColor(sensor_masked_image_bgr, cv2.COLOR_BGR2RGB)
        print("  Sensor overlay complete")

        return sensor_masked_image_np, combined_mask

    # =========================================================================
    # Stage 3: VLA Prediction
    # =========================================================================

    def predict_action(
        self,
        rgb_image: np.ndarray,
        sensor_masked_image: np.ndarray,
        instruction: str,
        predicted_sensor: str,
        object_description: str,
        state_full: np.ndarray,
        state_mask: np.ndarray = None,
        action_mask: np.ndarray = None,
        num_ddim_steps: int = 10,
        cfg_scale: float = 5.0,
        fov: torch.Tensor = None,
    ) -> np.ndarray:
        """
        Stage 3: VLA prediction for action generation.

        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            sensor_masked_image: Sensor masked image as numpy array (H, W, 3)
            instruction: Task instruction string
            predicted_sensor: Predicted sensor type
            object_description: Object description from VLM
            state_full: Robot state as numpy array (FULL_STATE_DIM,)
            state_mask: State mask - optional
            action_mask: Action mask - optional
            num_ddim_steps: Number of DDIM steps
            cfg_scale: Classifier-free guidance scale
            fov: Camera FOV tensor - optional

        Returns:
            Predicted action as numpy array (chunk_size, FULL_ACTION_DIM)
        """
        # Prepare instruction for VLA stage
        vla_instruction = (
            f"<image>Task: {instruction} Question: Choose from <None>, <thermal>, "
            "<radar>, <audio> for task. Tell the target object. Answer:"
        )

        # Prepare ground truth labels
        sensor_token_str = (
            predicted_sensor if predicted_sensor.startswith("<")
            else f"<{predicted_sensor}>"
        )
        ground_truth_text = (
            f"{sensor_token_str}. {object_description}"
        )

        # Process following PaddedCollatorForVLMAndVLA pattern
        tokenizer = self.model.processor.tokenizer

        original_add_eos = tokenizer.add_eos_token
        tokenizer.add_eos_token = False

        input_rgb_image = Image.fromarray(rgb_image)

        prompt_processed = self.model.processor(
            images=input_rgb_image,
            text=vla_instruction,
            return_tensors="pt",
            padding=False,
        )
        input_ids_prompt = prompt_processed['input_ids']
        prompt_len = input_ids_prompt.shape[1]

        answer_tokens = tokenizer(
            ground_truth_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        tokenizer.add_eos_token = original_add_eos

        eos_tensor = torch.tensor(
            [[tokenizer.eos_token_id]], dtype=answer_tokens['input_ids'].dtype
        )
        answer_with_eos = torch.cat([answer_tokens['input_ids'], eos_tensor], dim=1)
        answer_len = answer_with_eos.shape[1]

        seq_len = prompt_len + answer_len

        input_ids = torch.full((1, seq_len), tokenizer.pad_token_id, dtype=torch.long)
        input_ids[0, :prompt_len] = input_ids_prompt[0]

        labels = torch.full((1, seq_len), -100, dtype=torch.long)
        labels[0, prompt_len:seq_len] = answer_with_eos[0]

        attention_mask = torch.zeros((1, seq_len), dtype=torch.long)
        attention_mask[input_ids != tokenizer.pad_token_id] = 1
        attention_mask[labels != -100] = 1

        # Prepare dual images as [B, 2, C, H, W]
        rgb_pil = Image.fromarray(rgb_image)
        sensor_pil = Image.fromarray(sensor_masked_image)

        rgb_processed = self.model.processor(
            images=rgb_pil,
            text='<image>',
            return_tensors="pt",
        )
        rgb_pixel_values = rgb_processed['pixel_values']

        sensor_processed = self.model.processor(
            images=sensor_pil,
            text='<image>',
            return_tensors="pt",
        )
        sensor_pixel_values = sensor_processed['pixel_values']

        pixel_values = torch.stack(
            [rgb_pixel_values.squeeze(0), sensor_pixel_values.squeeze(0)], dim=0
        ).unsqueeze(0)

        # Prepare state and masks using subclass implementation
        unified_state, unified_state_mask, unified_action_mask, extra_data = \
            self.prepare_state_and_masks(state_full, state_mask, action_mask)

        # Transfer to human hand format
        human_state, human_state_mask, _, human_action_mask = transfer_xhand_to_human(
            unified_state, unified_state_mask, None, unified_action_mask
        )

        # Run VLA prediction
        norm_action = self.model.predict_action(
            image=pixel_values,
            instruction=input_ids,
            current_state=human_state.unsqueeze(0),
            current_state_mask=human_state_mask.unsqueeze(0),
            action_mask_torch=human_action_mask.unsqueeze(0),
            num_ddim_steps=num_ddim_steps,
            cfg_scale=cfg_scale,
            fov=fov,
            sample_times=1,
            task_mode="vlm_vla_cofinetune",
            attention_mask=attention_mask,
            labels=labels,
        )

        # Post-process action using subclass implementation
        norm_action = norm_action[0, :, :]
        unnorm_action = self.postprocess_action(norm_action, extra_data)

        return unnorm_action

    # =========================================================================
    # Complete Inference Flow
    # =========================================================================

    def inference(
        self,
        rgb_image: np.ndarray,
        instruction: str,
        state: np.ndarray,
        sensor_images: dict,
        use_vlm_cache: bool = None,
        use_rule_based: bool = False,
        num_ddim_steps: int = 10,
        cfg_scale: float = 5.0,
        save_debug_images: bool = True,
        debug: bool = False,
    ) -> np.ndarray:
        """
        Complete three-stage inference pipeline.

        Args:
            rgb_image: RGB image as numpy array (H, W, 3)
            instruction: Task instruction string
            state: Robot state as numpy array (STATE_DIM,) - right hand only
            sensor_images: dict with sensor images
            use_vlm_cache: Whether to use VLM cache (None = use default)
            use_rule_based: If True, use rule-based matching for sensor/object.
                           If False, use VLM prediction with rule-based fallback.
            num_ddim_steps: Number of DDIM steps
            cfg_scale: Classifier-free guidance scale
            save_debug_images: Whether to save debug images
            debug: If True, print detailed debug information

        Returns:
            Predicted action as numpy array (chunk_size, ACTION_DIM) - right hand only
        """
        print("\n" + "=" * 80)
        print("MuseVLA Inference Pipeline")
        print("=" * 80)

        # Create debug folder if saving images
        debug_folder = None
        if save_debug_images:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            debug_folder = os.path.join("./images", timestamp)
            os.makedirs(debug_folder, exist_ok=True)
            print(f"  Debug images will be saved to: {debug_folder}")

        # Convert input state to full state
        state_full = self.convert_input_state(state)

        # Stage 1: VLM prediction
        print("\n[Stage 1] Running VLM for object detection and sensor selection...")
        print(f"  Mode: {'rule-based' if use_rule_based else 'VLM-based'}")

        use_cache = use_vlm_cache if use_vlm_cache is not None else self.use_vlm_cache

        if use_cache and self.vlm_cache is not None:
            vlm_result = self.vlm_cache.get_or_compute(
                instruction,
                lambda: self.predict_vlm(rgb_image, instruction, use_rule_based=use_rule_based, debug=debug)
            )
            print(f"  VLM result obtained from cache: {vlm_result}")
        else:
            vlm_result = self.predict_vlm(rgb_image, instruction, use_rule_based=use_rule_based, debug=debug)
            print(f"  VLM result computed: {vlm_result}")

        predicted_sensor = vlm_result['predicted_sensor']
        object_description = vlm_result['object_description']
        source = vlm_result.get('source', 'unknown')

        print(f"  Predicted sensor: {predicted_sensor} (source: {source})")
        print(f"  Extracted object: {object_description}")
        print(f"  Raw VLM generated text: {vlm_result.get('raw_generated_text', '')}")

        # Stage 2: Generate sensor masked image
        print("\n[Stage 2] Generating sensor masked image using SAM3...")

        sensor_masked_image, combined_mask = self.generate_sensor_masked_image(
            rgb_image=rgb_image,
            object_description=object_description,
            predicted_sensor=predicted_sensor,
            sensor_images=sensor_images,
        )

        # Save debug images
        if save_debug_images and debug_folder:
            self._save_debug_info(
                debug_folder=debug_folder,
                rgb_image=rgb_image,
                sensor_images=sensor_images,
                combined_mask=combined_mask,
                sensor_masked_image=sensor_masked_image,
                predicted_sensor=predicted_sensor,
                object_description=object_description,
                instruction=instruction,
                state=state,
            )

        # Stage 3: VLA prediction
        print("\n[Stage 3] Running VLA for action prediction...")

        action_full = self.predict_action(
            rgb_image=rgb_image,
            sensor_masked_image=sensor_masked_image,
            instruction=instruction,
            predicted_sensor=predicted_sensor,
            object_description=object_description,
            state_full=state_full,
            num_ddim_steps=num_ddim_steps,
            cfg_scale=cfg_scale,
        )

        # Convert output action to per-hand format
        action_out = self.convert_output_action(action_full)

        print("\n" + "=" * 80)
        print("Inference Complete!")
        print(f"  Output shape: {action_out.shape}")
        print("=" * 80)

        return action_out

    def _save_debug_info(
        self,
        debug_folder: str,
        rgb_image: np.ndarray,
        sensor_images: dict,
        combined_mask: np.ndarray,
        sensor_masked_image: np.ndarray,
        predicted_sensor: str,
        object_description: str,
        instruction: str,
        state: np.ndarray,
    ):
        """Save debug images and state to the specified folder."""
        try:
            print(f"  Saving debug images to {debug_folder}...")

            Image.fromarray(rgb_image).save(
                os.path.join(debug_folder, "01_rgb_image.jpg")
            )

            if sensor_images.get('acoustic') is not None:
                Image.fromarray(sensor_images['acoustic']).save(
                    os.path.join(debug_folder, "02_sensor_acoustic.jpg")
                )
            if sensor_images.get('thermal_left') is not None:
                Image.fromarray(sensor_images['thermal_left']).save(
                    os.path.join(debug_folder, "02_sensor_thermal_left.jpg")
                )
            if sensor_images.get('thermal_right') is not None:
                Image.fromarray(sensor_images['thermal_right']).save(
                    os.path.join(debug_folder, "02_sensor_thermal_right.jpg")
                )
            if sensor_images.get('radar') is not None:
                Image.fromarray(sensor_images['radar']).save(
                    os.path.join(debug_folder, "02_sensor_radar.jpg")
                )

            mask_2d = combined_mask
            if mask_2d.ndim > 2:
                mask_2d = mask_2d.squeeze()
            if mask_2d.ndim == 2:
                mask_uint8 = (mask_2d.astype(np.uint8) * 255)
                Image.fromarray(mask_uint8, mode='L').save(
                    os.path.join(debug_folder, "03_sam3_mask.png")
                )

            if mask_2d.ndim == 2 and mask_2d.shape == rgb_image.shape[:2]:
                mask_overlay = rgb_image.copy()
                mask_overlay[mask_2d] = (
                    mask_overlay[mask_2d] * 0.5 +
                    np.array([255, 0, 0]) * 0.5
                ).astype(np.uint8)
                Image.fromarray(mask_overlay).save(
                    os.path.join(debug_folder, "04_mask_overlay.jpg")
                )

            Image.fromarray(sensor_masked_image).save(
                os.path.join(debug_folder, "05_sensor_masked_image.jpg")
            )

            # Save state to text file
            state_path = os.path.join(debug_folder, "state.txt")
            np.savetxt(state_path, state, fmt='%.6f')

            # Determine state format based on dimension
            state_dim = state.shape[0]
            if state_dim == 18:
                state_format = "6D_pos"
                state_format_description = "18D per hand: [6D EEF pose, 12D hand joints]"
            elif state_dim == 19:
                state_format = "7D_joint_angle"
                state_format_description = "19D per hand: [7D arm joints, 12D hand joints]"
            else:
                state_format = "unknown"
                state_format_description = f"Unknown format with {state_dim} dimensions"

            metadata = {
                "instruction": instruction,
                "predicted_sensor": predicted_sensor,
                "object_description": object_description,
                "mask_pixel_count": int(combined_mask.sum()),
                "mask_ratio": float(combined_mask.sum() / combined_mask.size),
                "sensor_images_provided": {
                    "acoustic": sensor_images.get('acoustic') is not None,
                    "thermal_left": sensor_images.get('thermal_left') is not None,
                    "thermal_right": sensor_images.get('thermal_right') is not None,
                    "radar": sensor_images.get('radar') is not None,
                },
                "state_format": {
                    "format": state_format,
                    "description": state_format_description,
                    "dimension": state_dim,
                },
            }
            with open(os.path.join(debug_folder, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"  Debug images saved successfully")

        except Exception as e:
            print(f"  Warning: Failed to save debug images: {e}")

    def clear_cache(self):
        """Clear VLM cache."""
        if self.vlm_cache:
            self.vlm_cache.clear()

    def set_sam3_confidence_threshold(self, threshold: float):
        """
        Set SAM3 segmentation confidence threshold at runtime.

        Args:
            threshold: Confidence threshold (0.0-1.0). Higher values result in
                      fewer but more confident detections.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        self.sam3_confidence_threshold = threshold
        self.sam3_processor.set_confidence_threshold(threshold)
        print(f"  SAM3 confidence threshold updated to: {threshold}")
