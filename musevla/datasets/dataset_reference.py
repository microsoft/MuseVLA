import numpy as np
import cv2
import os
import torch
from scipy.spatial.transform import Rotation as R
import bz2
import gzip
import lzma
import math
import pickle
from typing import Optional, Type
from prismatic.vla.datasets_pytorch.video_utils import load_video_decord, image_augmentation
from torch.utils.data import Dataset, Subset
import bisect
import os
from transformers import PreTrainedTokenizerBase
import json

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist
import random
from prismatic.vla.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES
import sys

from prismatic.vla.datasets_pytorch.augment_utils import contains_color_word, apply_color_augmentation

sys.path.append(os.path.dirname(__file__))

intrinsics = np.array([[643.43346676,   0.        , 649.93137192],
       [  0.        , 643.74326654, 368.75237164],
       [  0.        ,   0.        ,   1.        ]])

from functools import lru_cache

@lru_cache(maxsize=512)
def load_hand_file(hand_path):
    """
    Load entire hand file and cache it.
    
    Args:
        hand_path (str): Path to the hand.txt file.
    
    Returns:
        np.ndarray: Array of shape (num_lines, num_features) containing all hand data.
    """
    try:
        data = []
        with open(hand_path, 'r') as file:
            for line in file:
                data.append([float(x) for x in line.strip().split()])
        return np.array(data)
    except Exception as e:
        print(f"Error loading {hand_path}: {e}")
        return None

@lru_cache(maxsize=256)
def load_arm_file_npy(arm_path):
    """
    Load and cache .npy arm pose files.
    
    Args:
        arm_path (str): Path to the .npy file.
    
    Returns:
        np.ndarray: The loaded arm pose data.
    """
    return np.load(arm_path, allow_pickle=True)

class ActionFeature(object):
    HUMAN_LEFT_HAND = (0, 51)
    HUMAN_RIGHT_HAND = (51, 102)
    HUMAN_LEFT_TRANS = (0, 3)
    HUMAN_LEFT_ROT = (3, 6)
    HUMAN_LEFT_JOINTS = (6, 51)
    HUMAN_RIGHT_TRANS = (51, 54)
    HUMAN_RIGHT_ROT = (54, 57)
    HUMAN_RIGHT_JOINTS = (57, 102)
    
    ROBOT_LEFT_TRANS = (102, 105)
    ROBOT_LEFT_ROT_EULER = (105, 108)
    ROBOT_LEFT_JOINTS = (108, 115)
    ROBOT_LEFT_GRIPPER = (115, 116)
    ROBOT_LEFT_AGIBOT_6D_HAND = (116, 122)
    ROBOT_LEFT_XHAND = (122, 134)
    ROBOT_LEFT_ROT_6D = (134, 140)
    ROBOT_LEFT_ALL = (102, 147)
    
    ROBOT_RIGHT_TRANS = (147, 150)
    ROBOT_RIGHT_ROT_EULER = (150, 153)
    ROBOT_RIGHT_JOINTS = (153, 160)
    ROBOT_RIGHT_GRIPPER = (160, 161)
    ROBOT_RIGHT_AGIBOT_6D_HAND = (161, 167)
    ROBOT_RIGHT_XHAND = (167, 179)
    ROBOT_RIGHT_ROT_6D = (179, 185)
    ROBOT_RIGHT_ALL = (147, 192)
    
    PROGRESS = (191, 192)
    
    ALL_FEATURES = (0, 192)
    
    @classmethod
    def get_concatenated_action_feature_from_dict(cls, action_feature_dict):
        batch_size = next(iter(action_feature_dict.values())).shape[0]
        features = np.zeros((batch_size, cls.ALL_FEATURES[1]), dtype=np.float32)
        feature_mask = np.zeros((batch_size, cls.ALL_FEATURES[1]), dtype=bool)
        
        for key, value in action_feature_dict.items():
            assert len(value.shape) == 2
            start, end = getattr(cls, key)
            k = value.shape[1]
            features[:, start:start + k] = value
            feature_mask[:, start:start + k] = True
        return features, feature_mask
    
    @classmethod
    def get_dict_from_concatenated_action_feature(cls, feature, feature_mask):
        action_feature_dict = {}
        consts = {
            name: getattr(cls, name)
            for name in dir(cls)
            if name.isupper() and "ALL" not in name
        }
        for key, (start, end) in consts.items():
            k = np.sum(feature_mask[0, start:end])
            if k == 0:
                continue
            action_feature_dict[key] = feature[:, start:start + k]
        return action_feature_dict
    

class StateFeature(ActionFeature):
    ALL_FEATURES = (0, 212)
    HUMAN_LEFT_BETA = (192, 202)
    HUMAN_RIGHT_BETA = (202, 212)
    # CAMERA_FOV = (212, 214)

def compute_new_intrinsics_square(original_intrinsics, original_size, crop_size, resize_size):
    original_fx =original_intrinsics[0][0]
    original_fy =original_intrinsics[1][1]
    original_cx =original_intrinsics[0][2]
    original_cy =original_intrinsics[1][2]
    # Compute the crop offset (top-left corner of the crop)
    crop_offset = (original_size - crop_size) / 2
    # Update the principal point after the crop
    cropped_cx = original_cx - crop_offset
    cropped_cy = original_cy - crop_offset
    # Compute the scaling factor for resizing
    scale = resize_size / crop_size
    # Update the focal lengths and principal point after resizing
    new_fx = original_fx * scale
    new_fy = original_fy * scale
    new_cx = cropped_cx * scale
    new_cy = cropped_cy * scale
    intrinsics_matrix = np.array([
        [new_fx, 0, new_cx],
        [0, new_fy, new_cy],
        [0, 0, 1]
    ])
    return intrinsics_matrix

def square_center_crop(image):
    if len(image.shape) not in [2, 3]:
        raise ValueError("Input image must have shape (H, W) or (H, W, C).")
    h, w = image.shape[:2]
    if h > w:
        start_y = (h - w) // 2
        start_x = 0
    else:
        start_y = 0
        start_x = (w - h) // 2
    end_y = start_y + min(h, w)
    end_x = start_x + min(h, w)
    return image[start_y:end_y, start_x:end_x]

def square_crop(image, shift_right=50):
    if len(image.shape) not in [2, 3]:
      raise ValueError("Input image must have shape (H, W) or (H, W, C).")
    h, w = image.shape[:2]
    if h < w:
        start_x = w - h - shift_right
        if start_x < 0:
            raise ValueError("shift_right is too large for crop.")
        start_y = 0
    else:
        raise ValueError("Not Implemented yet.")
    end_y = start_y + min(h, w)
    end_x = start_x + min(h, w)
    return image[start_y:end_y, start_x:end_x]

def read_hand_line(hand_path, j):
    """
    Reads the j-th line from the hand.txt file using cached file data.

    Args:
        hand_path (str): Path to the hand.txt file.
        j (int): The line number to read (0-based index).

    Returns:
        np.ndarray or list: The values in the j-th line.
    """
    data = load_hand_file(hand_path)
    if data is None:
        return None
    if j < 0 or j >= len(data):
        print(f"Line {j} is out of bounds. The file has {len(data)} lines.")
        return None
    return data[j]

# def square_center_crop(image, crop_size):
#     if len(image.shape) not in [2, 3]:
#         raise ValueError("Input image must have shape (H, W) or (H, W, C).")
#     h, w = image.shape[:2]
#     if crop_size > h or crop_size > w:
#         raise ValueError("Crop size must be smaller than image size.")
#     start_y = (h - crop_size) // 2
#     start_x = (w - crop_size) // 2
#     end_y = start_y + crop_size
#     end_x = start_x + crop_size
    
#     return image[start_y:end_y, start_x:end_x]


class EpisodicDatasetCore(object):
    def __init__(self, root_dir):
        """Constructor.
        Args:
        root_dir: root_dir name. str.
        """
        self.root = root_dir
        with open(os.path.join(self.root, 'instructions_all.json'), 'r') as f:  
            self.meta = json.load(f) 
        # 初始化样本列表 
        self.action_length = 18
        self.samples = []
        # 遍历每个指令，并生成对应的样本  
        for instruction_id, item in enumerate(self.meta):  
            t1 = item['start']  
            t2 = item['end']  
            # 根据 t1 和 t2 生成样本  
            for frame in range(t1, t2):  
                self.samples.append((instruction_id, frame, t1, t2))

    def __len__(self):
        return len(self.samples)
    
    def get_lengths(self):
        return len(self.samples)

    def compute_Rt_in_cam_frame(self, R_t0, t_t0, global_orient_mat, transl):
        R_in_cam = np.dot(R_t0, global_orient_mat)
        t_in_cam = np.dot(R_t0, transl) + t_t0
        return R_in_cam, t_in_cam

    def get_neutral_action(self, action_length):
        neutral_action = np.zeros(action_length)
        return neutral_action

    def get_state(self, handpose, armpose):
        handpose = np.array(handpose)
        arm_state = np.array([armpose['x'], armpose['y'], armpose['z'], armpose['rx'], armpose['ry'], armpose['rz']])
        state = np.concatenate((arm_state, handpose), axis=0)
        return state

    def get_action(self, t0_frame, tgt_frame, instruction_id, episode_end, relative = True, relative_to_t0=False, absolute_joint=False, jump_frame=1):
    # def get_action(self, t0_frame, tgt_frame, instruction_id, episode_length, episode_handpose, episode_armpose, relative = True, relative_to_t0=False, absolute_joint=False):

        if tgt_frame > (episode_end - jump_frame):
            tgt_frame_next = tgt_frame # last frame of episode, set t+1 = t
        else:
            tgt_frame_next = tgt_frame + jump_frame

        valid_frame_flag = True

        relative_path = self.meta[instruction_id]['relative_path']
        hand_command_path = os.path.join(self.root, relative_path, 'ee_command','right_ee.txt')
        hand_path = os.path.join(self.root, relative_path, 'hand_status/right_hand_position.txt')
        arm_path = os.path.join(self.root, relative_path, '6d_pose/right_arm.npy')

        if tgt_frame_next == tgt_frame:
            tgt_handpose_command = read_hand_line(hand_command_path, tgt_frame_next)
        else:
            tgt_handpose_command = read_hand_line(hand_command_path, tgt_frame_next-1)

        arm_data = load_arm_file_npy(arm_path)
        tgt_next_armpose = arm_data[tgt_frame_next]

        # tgt_next_handpose = episode_handpose[tgt_frame_next]
        # tgt_next_armpose = episode_armpose[tgt_frame_next]
        tgt_next_arm_ee = np.array([tgt_next_armpose['x'], tgt_next_armpose['y'], tgt_next_armpose['z'], tgt_next_armpose['rx'], tgt_next_armpose['ry'], tgt_next_armpose['rz']])

        tgt_handpose_command = np.array(tgt_handpose_command)
        
        R_tgt_next = R.from_euler('xyz',tgt_next_arm_ee[3:6], degrees=False).as_matrix()
        t_tgt_next = tgt_next_arm_ee[:3]
        
        if relative_to_t0:
            t0_handpose = read_hand_line(hand_path, t0_frame)
            t0_armpose = arm_data[t0_frame]

            # t0_handpose = episode_handpose[t0_frame]
            # t0_armpose = episode_armpose[t0_frame]
            t0_arm_ee = np.array([t0_armpose['x'], t0_armpose['y'], t0_armpose['z'], t0_armpose['rx'], t0_armpose['ry'], t0_armpose['rz']])
            t0_handpose = np.array(t0_handpose)
            
            R_t0 = R.from_euler('xyz',t0_arm_ee[3:6], degrees=False).as_matrix()
            t_t0 = t0_arm_ee[:3]

            R_rel = np.dot(R_tgt_next, R_t0.T) # R_rel from t to t0 [3, 3]
            R_rel_euler= R.from_matrix(R_rel).as_euler('xyz', degrees=False) # [3]
            t_rel = t_tgt_next - t_t0 # t_rel from t to t0 [3]

            hand_pose_rel = tgt_handpose_command - t0_handpose

        else:
            tgt_handpose = read_hand_line(hand_path, tgt_frame)
            tgt_armpose = arm_data[tgt_frame]

            # tgt_handpose = episode_handpose[tgt_frame]
            # tgt_armpose = episode_armpose[tgt_frame]

            tgt_arm_ee = np.array([tgt_armpose['x'], tgt_armpose['y'], tgt_armpose['z'], tgt_armpose['rx'], tgt_armpose['ry'], tgt_armpose['rz']])
            tgt_handpose = np.array(tgt_handpose)
            
            R_tgt = R.from_euler('xyz',tgt_arm_ee[3:6], degrees=False).as_matrix()
            t_tgt = tgt_arm_ee[:3]
            R_rel = np.dot(R_tgt_next, R_tgt.T) # R_rel from t to t+1 [3, 3]
            R_rel_euler= R.from_matrix(R_rel).as_euler('xyz', degrees=False) # [3]
            t_rel = t_tgt_next - t_tgt # t_rel from t to t+1 [3]  
            hand_pose_rel = tgt_handpose_command - tgt_handpose  

        if relative:
            if absolute_joint:
                action = np.concatenate([t_rel, R_rel_euler, tgt_handpose_command])
            else:   
                action = np.concatenate([t_rel, R_rel_euler, hand_pose_rel])
        else:
            R_tgt_next_euler = R.from_matrix(R_tgt_next).as_euler('xyz', degrees=False)
            action = np.concatenate([t_tgt_next, R_tgt_next_euler, tgt_handpose_command])
        return action, valid_frame_flag
    

    def get_item_frame(self, idx, image_past_window_size=0, image_future_window_size=0, action_past_window_size=0, 
        action_future_window_size=0, relative=True, relative_to_t0=False, absolute_joint=True, jump_frame=5, adaptive_jump_frame=2, threshold=0.087):
        instruction_id, frame_id, start, end = self.samples[idx]
        episode_length = end - start + 1
        relative_path = self.meta[instruction_id]['relative_path']

        image_path = os.path.join(self.root, relative_path, 'images', f'{frame_id:05d}.jpg')
        image_list = []
        if image_past_window_size == 0 and image_future_window_size == 0:
            try:
                image = cv2.imread(image_path)[:, :, ::-1]
                image_list.append(image)
            except:
                raise ValueError(f"Failed to load image {image_path}")
        else:
            raise ValueError("image_past_window_size and image_future_window_size should be 0")


        image_mask = [True] * len(image_list)
        hand_path = os.path.join(self.root, relative_path, 'hand_status/right_hand_position.txt')
        arm_path = os.path.join(self.root, relative_path, '6d_pose/right_arm.npy')

        handpose = read_hand_line(hand_path, frame_id)
        arm_data = load_arm_file_npy(arm_path)
        armpose = arm_data[frame_id]

        instruction = self.meta[instruction_id]['instruction']
        instruction = instruction.strip()
        instruction = f"Left: None. Right: {instruction}"

        # color augmentation
        # if contains_color_word(instruction):
        #     preserve_hue = True
        # else:
        #     preserve_hue = False
        preserve_hue = False
        image_list = [apply_color_augmentation(image_list[0], preserve_hue=preserve_hue)]
        
        # if instruction.strip() == "Pick up the tomato and place it into the white basket.":
        #     instruction = "picks up the tomato and place it into the white basket."
        action_list = []
        action_mask = []
        current_state = []
        current_beta = []
        current_state_mask = []

        state_tmp = self.get_state(handpose, armpose)
        action_list_tmp = []
        action_mask_tmp = []
        hand_command_path = os.path.join(self.root, relative_path, 'ee_command','right_ee.txt')
        # 目标采样数量
        target_samples = action_past_window_size + action_future_window_size + 1
        current_jump_frame = jump_frame
        threshold = threshold
        # 第一次尝试：使用原始步长
        for i in range(frame_id - (jump_frame * action_past_window_size), frame_id + jump_frame * (action_future_window_size + 1) , jump_frame):
            max_l1  = 0
            if i == frame_id - (jump_frame * action_past_window_size):
                if i - 1 > start: 
                    prev_handpose_command = read_hand_line(hand_command_path, i - 1)
                    tgt_handpose_command = read_hand_line(hand_command_path, i)
                    diff = np.array(prev_handpose_command) - np.array(tgt_handpose_command)
                    max_l1 = np.max(np.abs(diff))
                else:
                    max_l1 = 0
            if i < start:
                action = self.get_neutral_action(self.action_length)
                # not 0, if delta Euler (I in Rotation)
                action_list_tmp.append(action)
                action_mask_tmp.append(False)
            elif i >= (1 + end):
                # last_frame_idx = end + 1 - i - 1
                last_frame_idx = (end + 1 - i - 1) // jump_frame # -((i - 1 - end) // jump_frame) - 1
                last_frame_validity = action_mask_tmp[last_frame_idx]
                if (relative is True) and (relative_to_t0 is False):
                    action = self.get_neutral_action(self.action_length)
                    if absolute_joint:
                        last_frame_action = action_list_tmp[last_frame_idx]
                        action[6:18] = last_frame_action[6:18]
                else:
                    last_frame_action = action_list_tmp[last_frame_idx]
                    action = last_frame_action
                action_list_tmp.append(action)
                mask = ((i - end - 1) < 8 * jump_frame) & last_frame_validity # last_frame_validity
                action_mask_tmp.append(mask)
            else:
                action, valid_frame_flag = self.get_action(frame_id, i, instruction_id, end, relative=relative, relative_to_t0=relative_to_t0, absolute_joint=absolute_joint, jump_frame = jump_frame)
                
                # 检查max_l1条件
                if len(action_list_tmp) > 0 or max_l1 > 0:
                    if max_l1 == 0:
                        prev_action = action_list_tmp[-1]
                        # 计算后12维的max_l1 = np.max(np.abs(diff))
                        l1_diff = np.abs(action[6:18] - prev_action[6:18])
                        max_l1 = np.max(l1_diff)

                    if max_l1 > threshold:
                        # max_l1超过阈值，从上一步开始重新采样，使用adaptive_jump_frame
                        # 本次开始重新采样，使用可调步长继续采样剩余的frames
                        remaining_samples = target_samples - len(action_list_tmp)
                        current_i = i
                        
                        for _ in range(remaining_samples):
                            if current_i < start:
                                action_new = self.get_neutral_action(self.action_length)
                                action_list_tmp.append(action_new)
                                action_mask_tmp.append(False)
                            elif current_i >= (1 + end):
                                last_frame_idx_new = max(len(action_list_tmp) - 1, 0)
                                if last_frame_idx_new >= 0:
                                    last_frame_validity_new = action_mask_tmp[last_frame_idx_new]
                                else:
                                    last_frame_validity_new = False
                                    
                                if (relative is True) and (relative_to_t0 is False):
                                    action_new = self.get_neutral_action(self.action_length)
                                    if absolute_joint and last_frame_idx_new >= 0:
                                        last_frame_action_new = action_list_tmp[last_frame_idx_new]
                                        action_new[6:18] = last_frame_action_new[6:18]
                                else:
                                    if last_frame_idx_new >= 0:
                                        action_new = action_list_tmp[last_frame_idx_new]
                                    else:
                                        action_new = self.get_neutral_action(self.action_length)
                                        
                                action_list_tmp.append(action_new)
                                mask_new = mask = ((current_i - end - 1) < 8 * adaptive_jump_frame) & last_frame_validity_new #last_frame_validity_new
                                action_mask_tmp.append(mask_new)
                            else:
                                action_new, valid_frame_flag_new = self.get_action(frame_id, current_i, instruction_id, end, relative=relative, relative_to_t0=relative_to_t0, absolute_joint=absolute_joint, jump_frame=adaptive_jump_frame)
                                action_list_tmp.append(action_new)
                                action_mask_tmp.append(valid_frame_flag_new)
                            
                            current_i += adaptive_jump_frame  # 使用可调步长
                        
                        break  # 退出原循环
                
                action_list_tmp.append(action)
                action_mask_tmp.append(valid_frame_flag)

        # if len(action_list_tmp) < target_samples:
        #     print(f"Error: Unable to generate enough samples. Current samples: {len(action_list_tmp)}, Target samples: {target_samples}")
        #     # 确保最终达到目标采样数量
        #     while len(action_list_tmp) < target_samples:
        #         action_list_tmp.append(self.get_neutral_action(self.action_length))
        #         action_mask_tmp.append(False)
            
        #     # 如果超过目标数量，截取到目标数量
        #     if len(action_list_tmp) > target_samples:
        #         action_list_tmp = action_list_tmp[:target_samples]
        #         action_mask_tmp = action_mask_tmp[:target_samples]

        action_list_tmp = np.stack(action_list_tmp)
        action_mask_tmp = np.stack(action_mask_tmp)
        action_mask_tmp = action_mask_tmp[..., np.newaxis]
        # padding
        padding_state = np.zeros_like(state_tmp)
        padding_actions = np.zeros_like(action_list_tmp)
        zero_mask = np.zeros_like(action_mask_tmp, dtype=bool)

        ### right hand, padding the left hand and arm ###
        current_state.append(padding_state)
        current_state_mask.append(False)
        current_state.append(state_tmp)
        current_state_mask.append(True)
        
        action_list.append(padding_actions)
        action_mask.append(zero_mask)
        action_list.append(action_list_tmp)
        action_mask.append(action_mask_tmp)

        action_list = np.concatenate(action_list,axis=1) # [T, act_len * 2] 
        action_mask = np.concatenate(action_mask,axis=1) # [T, 2] 2: left/right 
        current_state = np.concatenate(current_state,axis=0) # [(act_len + 10) * 2]
        current_state_mask = np.stack(current_state_mask) # [2]
        # breakpoint()
        ori_w = image_list[0].shape[1]
        ori_h = image_list[0].shape[0]
        # ori_w = intrinsics[0][2] * 2  # original width
        # ori_h = intrinsics[1][2] * 2  # original height

        hfov = 2 * np.arctan(ori_w / (2 * intrinsics[0][0]))  # fx is the horizontal focal length
        vfov = 2 * np.arctan(ori_h / (2 * intrinsics[1][1]))  # fy is the vertical focal length

        fov = np.array([hfov, vfov])  # (2,)

        return {
            "instruction": instruction,
            "image_list": image_list,
            "action_list": action_list,
            "image_mask": image_mask,
            "action_mask": action_mask,
            "current_state": current_state,
            "current_state_mask": current_state_mask,
            "fov": fov,
        }

class FrameDataset(Dataset):
    def __init__(self, dataset_folder, dataset_name,
                 text_tokenizer: PreTrainedTokenizerBase, image_transform: ImageTransform, prompt_builder_fn: Type[PromptBuilder],
                 image_past_window_size=0, image_future_window_size=0, action_past_window_size=0, action_future_window_size=0,
                 image_augmentation=False, normalization=True, predict_stop_token=False, processor=None, jump_frame=5, adaptive_jump_frame=2, angle_align=False, is_oxe=False, statistics_file="teledata_omni_0123_handcommand_absolute_joint_valid184_jump4_ada2_maxl1.json"):
        # only support image_past_window_size=0 now (in the post transform)
        """Both past and future window size does not include the current frame"""
        self.episodic_dataset_core = EpisodicDatasetCore(dataset_folder)
        self._length = self.episodic_dataset_core.get_lengths()
        # self._length = sum(self.episodic_lengths)
        # self._accumulate_lengths = [0]
        # for length in self.episodic_lengths:
        #     self._accumulate_lengths.append(self._accumulate_lengths[-1] + length)
        # self._length = sum(self.episodic_lengths)
        self.image_past_window_size = image_past_window_size
        self.image_future_window_size = image_future_window_size
        self.action_past_window_size = action_past_window_size
        self.action_future_window_size = action_future_window_size
        self.jump_frame = jump_frame
        self.adaptive_jump_frame = adaptive_jump_frame
        self.dataset_name = dataset_name
        self.text_tokenizer = text_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.image_augmentation = image_augmentation
        self.normalization = normalization
        with open(os.path.join(dataset_folder, statistics_file), 'r') as file:  
            self.statistics = json.load(file)
        if self.normalization:
            ### right hand only now
            self.data_statistics = {
                'state_right_mean': np.array(self.statistics['state_right']['mean']),
                'state_right_std': np.array(self.statistics['state_right']['std']),
                'action_right_mean': np.array(self.statistics['action_right']['mean']),
                'action_right_std': np.array(self.statistics['action_right']['std']),
            }
        self.processor = processor
        self.predict_stop_token = predict_stop_token
        self.angle_align = angle_align
        self.is_oxe = is_oxe

    def __len__(self):
        return self._length
    
    def __getitem__(self, idx):
        # check inter/union/none
        # self.episodic_dataset_core.correspondence_union[idx]
        # episode_id = bisect.bisect_right(self._accumulate_lengths, idx) - 1
        # frame_id = idx - self._accumulate_lengths[episode_id]
        # episode_id_bf, frame_id_bf = self.find_index_bf(idx)
        # assert 0 <= frame_id < self.episodic_lengths[episode_id]
        # assert episode_id == episode_id_bf
        # assert frame_id == frame_id_bf
        data = self.episodic_dataset_core.get_item_frame(
            idx, 
            action_past_window_size=self.action_past_window_size, 
            action_future_window_size=self.action_future_window_size,
            image_past_window_size=self.image_past_window_size,
            image_future_window_size=self.image_future_window_size,
            jump_frame = self.jump_frame,
            adaptive_jump_frame = self.adaptive_jump_frame
        )
        return self.post_transform(data)
    
    def post_transform(self, data):
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        img = data["image_list"][-1]
        #check
        img = Image.fromarray(img)
        # To do: not support image augmentation now
        if self.image_augmentation:
            img = image_augmentation([img])[0]
        # pixel_values = self.image_transform(img)
        lang = data["instruction"]
        imgs = []
        imgs.append(img)
        # lang = rlds_batch["task"]["language_instruction"].decode().lower()
        # for raw in rlds_batch["observation"]["image_primary"]:
        #     img = Image.fromarray(raw)
        #     imgs.append(img)
        # sampled_imgs = imgs[::3] 
        #[img0, img1, img2, img3, img4, img5, img6(current observation)] --> [img0, img3, img6(current observation)]
        
        #lang = '<image>'*len(sampled_imgs) + lang
        lang = '<image>'*len(imgs) + lang
        model_inputs = self.processor(text=lang, images=imgs, return_tensors="pt").to(torch.float32)
        input_ids = model_inputs["input_ids"]
        pixel_values = model_inputs["pixel_values"]
        input_ids = input_ids.squeeze(0) 
        # pixel_values = pixel_values.squeeze(0)
        # pixel_values.shape need to be [num_img, 3, 224, 224]

        fov = torch.tensor(data["fov"], dtype=torch.float32)
        action_np = np.asarray(data["action_list"])
        state_np = np.asarray(data["current_state"])

        if self.normalization:
            action_np[:,18:] = (action_np[:,18:] - self.data_statistics['action_right_mean']) / (self.data_statistics['action_right_std'] + 1e-7)
            state_np[18:] = (state_np[18:] - self.data_statistics['state_right_mean']) / (self.data_statistics['state_right_std'] + 1e-7)

        actions_list = torch.tensor(action_np, dtype=torch.float32)
        action_mask = torch.tensor(np.asarray(data["action_mask"]), dtype=torch.bool)

        image_mask = torch.tensor(np.asarray(data["image_mask"]), dtype=torch.bool)
        
        current_state = torch.tensor(state_np, dtype=torch.float32)
        
        current_state_mask = torch.tensor(np.asarray(data["current_state_mask"]), dtype=torch.bool)
        expanded_state_mask = current_state_mask.repeat_interleave(18) # current_state.shape[0] / 2 = 61  
        current_state = current_state * expanded_state_mask # padding 0 for invalid state dim [36]
        
        unified_state = torch.zeros(212, dtype=torch.float32)
        unified_state_mask = torch.zeros(212, dtype=torch.bool)



        # padding_state_zeros = torch.zeros(90, dtype=torch.bool)
        # current_state = torch.cat((current_state, padding_state_zeros.to(torch.float32)), dim=0) # [122+90]

        # current_state_mask = torch.cat((expanded_state_mask, padding_state_zeros), dim=0) # [122+90]
        # current_state = torch.cat((current_state, expanded_state_mask.to(torch.float32), padding_zeros), dim=0)
        
        # expanded_action_mask = action_mask.repeat_interleave(61)   
        # expand the mask
        mask_0_18 = action_mask[:, 0].unsqueeze(1).expand(-1, 18)  # current_state.shape[1] / 2 = 51
        mask_18_36 = action_mask[:, 1].unsqueeze(1).expand(-1, 18)
        expanded_action_mask = torch.cat((mask_0_18, mask_18_36), dim=1)
        actions = actions_list * expanded_action_mask # padding 0 for invalid action dim

        unified_actions = torch.zeros((actions.shape[0], 192), dtype=torch.float32)
        unified_action_mask = torch.zeros((expanded_action_mask.shape[0], 192), dtype=torch.bool)
        if self.is_oxe:
            unified_state[0:6] = current_state[18:24]
            unified_state_mask[0:6] = expanded_state_mask[18:24]
            unified_actions[:, 0:6] = actions[:, 18:24]
            unified_action_mask[:, 0:6] = expanded_action_mask[:, 18:24]
        else:
            unified_state[51:57] = current_state[18:24]
            unified_state_mask[51:57] = expanded_state_mask[18:24]
            unified_actions[:, 51:57] = actions[:, 18:24]
            unified_action_mask[:, 51:57] = expanded_action_mask[:, 18:24]

        if self.angle_align:
            mapping = [
                (28, 59, 1),
                (29, 65, 1),
                (30, 68, 1),
                (31, 74, 1),
                (34, 77, 1),
                (35, 83, 1),
                (32, 86, 1),
                (33, 92, 1),
                (25, 94, -1),
                (24, 95, 1),
                (26, 97, -1),
                (27, 58, 1),
            ]

            for src, dst, sign in mapping:
                unified_state[dst] = sign * current_state[src]
                unified_actions[:, dst] = sign * actions[:, src]
                unified_action_mask[:, dst] = expanded_action_mask[:, src]
            unified_state_mask[51:102] = 1
        else:
            unified_state[167:179] = current_state[24:36]
            unified_state_mask[167:179] = expanded_state_mask[24:36]

            unified_actions[:, 167:179] = actions[:, 24:36]
            unified_action_mask[:, 167:179] = expanded_action_mask[:, 24:36]
        # padding_actions_zeros = torch.zeros((actions_list.shape[0], 90), dtype=torch.bool)
        # actions = torch.cat((actions, padding_actions_zeros.to(torch.float32)), dim=1) # [T, (102+90)]
        # action_mask = torch.cat((expanded_action_mask, padding_actions_zeros), dim=1) # [T, (102+90)]


        # past_action_window_size = len(imgs)-1
        # action = action[past_action_window_size:]
        # action = torch.tensor(action, dtype=torch.float32)
        # action_mask = None
        # if "action_mask" in rlds_batch:
        #     action_mask = torch.tensor(rlds_batch["action_mask"], dtype=torch.bool)

        labels = None
        # # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        # labels[: -1] = IGNORE_INDEX
        # if not self.predict_stop_token:
        #     labels[-1] = IGNORE_INDEX

        return_dict = dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            dataset_name=self.dataset_name,
            actions=unified_actions,
            action_masks=unified_action_mask,
            current_state_mask=unified_state_mask,
            current_state=unified_state,
            fov = fov
        )

        return return_dict

    
    def find_index_bf(self, idx):
        episode_id = 0
        for i in range(len(self.episodic_lengths)):
            if idx < self.episodic_lengths[i]:
                episode_id = i
                break
            idx -= self.episodic_lengths[i]
        return episode_id, idx

    # this function is merged to __init__
    # def add(self, annotation_file): 
    #     annotation_dict = np.load(annotation_file, allow_pickle=True).item()
    #     self.annotation = annotation_dict['annotation']
    #     self.correspondence = annotation_dict['correspondence']
    #     self.filtered_correspondence = annotation_dict['filtered_correspondence']
    #     self.num_valid_frames = len(self.filtered_correspondence)
    #     self.num_total_frames = len(self.correspondence)
    #     self.num_episodes = len(self.annotation['s_kept_frames'])

class MultipleWeightedDataset(Dataset):
    def __init__(self, datasets, weights=None):
        self.datasets = datasets
        if weights is None:
            weights = [1] * len(datasets)
        self.weights = weights
        self._length = sum([len(dataset) for dataset in datasets])
        self._accumulate_lengths = [0]
        for dataset in datasets:
            self._accumulate_lengths.append(self._accumulate_lengths[-1] + len(dataset))
    
    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        if isinstance(index, int):
            dataset_id = bisect.bisect_right(self._accumulate_lengths, index) - 1
            idx = index - self._accumulate_lengths[dataset_id]
            return self.datasets[dataset_id][idx]
        elif isinstance(index, tuple):
            dataset_id, idx = index
            return self.datasets[dataset_id][idx]

    @classmethod
    def load_datasets(cls, dataset_folder, data_mix,
                      text_tokenizer: PreTrainedTokenizerBase, image_transform: ImageTransform, prompt_builder_fn: Type[PromptBuilder],
                      image_past_window_size=0, image_future_window_size=0, action_past_window_size=0, action_future_window_size=0,
                      image_augmentation=False):
        dataset_weight_list = []
        if data_mix in OXE_NAMED_MIXTURES:
            dataset_weight_list = OXE_NAMED_MIXTURES[data_mix]
        else:
            dataset_weight_list = [(data_mix, 1)]
        
        datasets = []
        weights = []
        for dataset_name, weight in dataset_weight_list:
            print("Loading dataset:", dataset_name)
            dataset = FrameDataset(os.path.join(dataset_folder, dataset_name), dataset_name, 
                                   text_tokenizer, image_transform, prompt_builder_fn, 
                                   image_past_window_size, image_future_window_size, 
                                   action_past_window_size, action_future_window_size,
                                   image_augmentation, jump_frame=jump_frame, adaptive_jump_frame=adaptive_jump_frame, angle_align=angle_align, is_oxe=is_oxe)
            datasets.append(dataset)
            weights.append(weight)
        return cls(datasets, weights)

    @classmethod
    def load_single_dataset(cls, dataset_folder, data_mix,
                      text_tokenizer: PreTrainedTokenizerBase, image_transform: ImageTransform, prompt_builder_fn: Type[PromptBuilder],
                      image_past_window_size=0, image_future_window_size=0, action_past_window_size=0, action_future_window_size=0,
                      image_augmentation=False, normalization=True, predict_stop_token = False, processor = None, data_slice=None, jump_frame=5, adaptive_jump_frame=2, angle_align=False, is_oxe=False):
        dataset_weight_list = []
        if data_mix in OXE_NAMED_MIXTURES:
            dataset_weight_list = OXE_NAMED_MIXTURES[data_mix]
        else:
            dataset_weight_list = [(data_mix, 1)]
        assert len(dataset_weight_list) == 1, f"Expected one dataset only, but got {len(dataset_weight_list)} elements."  
        datasets = []
        weights = []
        statistics_file = 'teledata_omni_0123_handcommand_absolute_joint_valid184_jump4_ada2_maxl1.json'
        # if jump_frame == 5:
        #     statistics_file = 'teledata_08data_0908label_handcommand_absolute_joint_valid184_jump5_ada2_maxl1.json'
        # elif jump_frame == 4:
        #     statistics_file = 'teledata_08data_0908label_handcommand_absolute_joint_valid184_jump4_ada2_maxl1.json'
        # elif jump_frame == 3:
        #     statistics_file = 'teledata_08data_0908label_handcommand_absolute_joint_valid184_jump3_ada2_maxl1.json'
        for dataset_name, weight in dataset_weight_list:
            print("Loading dataset:", dataset_name)
            dataset = FrameDataset(os.path.join(dataset_folder, dataset_name), dataset_name, 
                                   text_tokenizer, image_transform, prompt_builder_fn, 
                                   image_past_window_size, image_future_window_size, 
                                   action_past_window_size, action_future_window_size,
                                   image_augmentation, normalization, predict_stop_token, processor, jump_frame=jump_frame, adaptive_jump_frame=adaptive_jump_frame, angle_align=angle_align, is_oxe=is_oxe, statistics_file=statistics_file)
            # subset_indices = data_slice
            # subset_dataset = Subset(dataset, subset_indices)  
            datasets.append(dataset)
            weights.append(weight)
        return cls(datasets, weights)

# class MultipleDatasetWeightedDistributedBatchSampler(torch.utils.data.DistributedSampler):
class MultipleDatasetWeightedDistributedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, 
                 dataset, batch_size,  
                 drop_last: bool = False, 
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 0):
        self.dataset = dataset
        self.epoch = 0
        self.step = 0
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")
        self.num_replicas = num_replicas
        self.rank = rank
        print(f"Creating Distributed Batch Sampler with rank {rank} and num_replicas {num_replicas}")
        self.prepare_sample_size()
    
    def prepare_sample_size(self):
        self._dataset_lengths = [len(dataset) for dataset in self.dataset.datasets]
        self.weights = self.dataset.weights
        self._sample_size = [int(weight * length) for weight, length in zip(self.weights, self._dataset_lengths)]
        self.total_size = sum(self._sample_size)
        iter_size = self.batch_size * self.num_replicas
        if self.drop_last:
            self.num_iters = self.total_size // iter_size
        else:
            self.num_iters = (self.total_size + iter_size - 1) // iter_size
        self.num_samples = self.num_iters * iter_size
    
    def create_indices(self, dataset_id, epoch):
        indices = list(range(self._dataset_lengths[dataset_id]))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed((self.seed + epoch) * len(self.dataset.datasets) + dataset_id)
            indices = torch.randperm(len(indices), generator=g).tolist()
        return indices
    
    def create_indices_range(self, dataset_id, start, end):
        dataset_length = self._dataset_lengths[dataset_id]
        start_epoch = start // dataset_length
        start_idx = start % dataset_length
        end_epoch = end // dataset_length
        end_idx = end % dataset_length
        if end_idx == 0:
            end_epoch -= 1
            end_idx = dataset_length
        assert start_epoch <= end_epoch
        indices = []
        for epoch in range(start_epoch, end_epoch + 1):
            epoch_indices = self.create_indices(dataset_id, epoch)
            if epoch == start_epoch and epoch == end_epoch:
                epoch_indices = epoch_indices[start_idx:end_idx]
            elif epoch == start_epoch:
                epoch_indices = epoch_indices[start_idx:]
            elif epoch == end_epoch:
                epoch_indices = epoch_indices[:end_idx]
            else:
                epoch_indices = epoch_indices
            indices.extend(epoch_indices)
        assert len(indices) == end - start
        return indices
    
    def shuffle_dataset_indices(self, indices):
        dataset_id = []
        for i in range(len(indices)):
            dataset_id.extend([i] * len(indices[i]))
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(dataset_id)
        dataset_index_list = []
        dataset_count = [0] * len(self.dataset.datasets)
        for i in range(len(dataset_id)):
            di = dataset_id[i]
            si = indices[di][dataset_count[di]]
            dataset_index_list.append((di, si))
            dataset_count[di] += 1
        for i in range(len(dataset_count)):
            assert dataset_count[i] == len(indices[i])
        return dataset_index_list
            

    def prepare_indices(self):
        indices = []
        for i in range(len(self.dataset.datasets)):
            start = self.epoch * self._sample_size[i]
            end = (self.epoch + 1) * self._sample_size[i]
            indices.append(self.create_indices_range(i, start, end))
        dataset_index_list = self.shuffle_dataset_indices(indices)
        return dataset_index_list
    
    def dataset_statistics(self, dataset_index_list):
        dataset_count = [0] * len(self.dataset.datasets)
        for di, si in dataset_index_list:
            dataset_count[di] += 1
        s = sum(dataset_count)
        for i in range(len(dataset_count)):
            print(f"Dataset {i} count: {dataset_count[i]}, ratio: {dataset_count[i] / s:.4f}")

    def __iter__(self):
        dataset_index_list = self.prepare_indices()

        if not self.drop_last:
            padding_size = self.num_samples - len(dataset_index_list)
            if padding_size <= len(dataset_index_list):
                dataset_index_list += dataset_index_list[:padding_size]
            else:
                dataset_index_list += (dataset_index_list * math.ceil(padding_size / len(dataset_index_list)))[:padding_size]
        else:
            dataset_index_list = dataset_index_list[:self.num_samples]
        assert len(dataset_index_list) == self.num_samples

        dataset_index_list = dataset_index_list[self.rank:self.num_samples:self.num_replicas]
        assert len(dataset_index_list) == self.num_iters * self.batch_size

        self.dataset_statistics(dataset_index_list)
        # print("Samples:", dataset_index_list[:200])

        for i in range(self.step, self.num_iters):
            # print("Iterating", i, dataset_index_list[i * self.batch_size:(i + 1) * self.batch_size])
            yield dataset_index_list[i * self.batch_size:(i + 1) * self.batch_size]
        self.set_epoch(self.epoch + 1)
        print("Epoch", self.epoch, "completed")

    def __len__(self):
        return self.num_iters

    def set_epoch(self, epoch, step=0):
        assert epoch >= 0
        step = step % self.num_iters
        if not (0 <= step < self.num_iters):
            print(f"Assertion failed: step={step}, self.num_iters={self.num_iters}, epoch={epoch}")
            raise ValueError("Step is out of valid range.")
        self.epoch = epoch
        self.step = step

# @dataclass
# class HandBatchTransformPaligemma2:
#     predict_stop_token: bool = True
#     processor:PaliGemmaProcessor = None
#     def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
#         """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
#         img = data["image_list"][-1]
#         #check
#         img = Image.fromarray(img)
#         # To do: not support image augmentation now
#         if self.image_augmentation:
#             img = image_augmentation([img])[0]
#         # pixel_values = self.image_transform(img)
#         imgs = []
#         imgs.append(img)
#         lang = data["instruction"]
#         ###############################
#         if rlds_batch["action"].shape[0] > 1:
#             dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"]
#         else:
#             dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]

#         # lang = rlds_batch["task"]["language_instruction"].decode().lower()
#         # for raw in rlds_batch["observation"]["image_primary"]:
#         #     img = Image.fromarray(raw)
#         #     imgs.append(img)
#         # sampled_imgs = imgs[::3] 
#         #[img0, img1, img2, img3, img4, img5, img6(current observation)] --> [img0, img3, img6(current observation)]
        
#         #lang = '<image>'*len(sampled_imgs) + lang
#         lang = '<image>'*len(imgs) + lang
#         model_inputs = self.processor(text=lang, images=imgs, return_tensors="pt").to(torch.float32)
#         input_ids = model_inputs["input_ids"]
#         pixel_values = model_inputs["pixel_values"]
#         input_ids = input_ids.squeeze(0) 
#         # pixel_values = pixel_values.squeeze(0)
#         # pixel_values.shape need to be [num_img, 3, 224, 224]

#         action_np = np.asarray(data["action_list"])
#         state_np = np.asarray(data["current_state"])

#         if self.normalization:
#             action_np[:,:51] = (action_np[:,:51] - self.data_statistics['action_left_mean']) / (self.data_statistics['action_left_std'] + 1e-7)
#             action_np[:,51:] = (action_np[:,51:] - self.data_statistics['action_right_mean']) / (self.data_statistics['action_right_std'] + 1e-7)
#             state_np[:61] = (state_np[:61] - self.data_statistics['state_left_mean']) / (self.data_statistics['state_left_std'] + 1e-7)
#             state_np[61:] = (state_np[61:] - self.data_statistics['state_right_mean']) / (self.data_statistics['state_right_std'] + 1e-7)

#         actions_list = torch.tensor(action_np, dtype=torch.float32)
#         action_mask = torch.tensor(np.asarray(data["action_mask"]), dtype=torch.bool)

#         image_mask = torch.tensor(np.asarray(data["image_mask"]), dtype=torch.bool)
        
#         current_state = torch.tensor(state_np, dtype=torch.float32)
        
#         current_state_mask = torch.tensor(np.asarray(data["current_state_mask"]), dtype=torch.bool)
#         expanded_state_mask = current_state_mask.repeat_interleave(61) # current_state.shape[0] / 2 = 61  
#         current_state = current_state * expanded_state_mask # padding 0 for invalid state dim
#         padding_state_zeros = torch.zeros(90, dtype=torch.bool)
#         current_state = torch.cat((current_state, padding_state_zeros.to(torch.float32)), dim=0) # [122+90]

#         current_state_mask = torch.cat((expanded_state_mask, padding_state_zeros), dim=0) # [122+90]
#         # current_state = torch.cat((current_state, expanded_state_mask.to(torch.float32), padding_zeros), dim=0)
        
#         # expanded_action_mask = action_mask.repeat_interleave(61)   
#         # expand the mask
#         mask_0_51 = action_mask[:, 0].unsqueeze(1).expand(-1, 51)  # current_state.shape[1] / 2 = 51
#         mask_51_102 = action_mask[:, 1].unsqueeze(1).expand(-1, 51)
#         expanded_action_mask = torch.cat((mask_0_51, mask_51_102), dim=1)
#         actions = actions_list * expanded_action_mask # padding 0 for invalid action dim

#         padding_actions_zeros = torch.zeros((actions_list.shape[0], 90), dtype=torch.bool)
#         actions = torch.cat((actions, padding_actions_zeros.to(torch.float32)), dim=1) # [T, (102+90)]
#         action_mask = torch.cat((expanded_action_mask, padding_actions_zeros), dim=1) # [T, (102+90)]


#         # past_action_window_size = len(imgs)-1
#         # action = action[past_action_window_size:]
#         # action = torch.tensor(action, dtype=torch.float32)
#         # action_mask = None
#         # if "action_mask" in rlds_batch:
#         #     action_mask = torch.tensor(rlds_batch["action_mask"], dtype=torch.bool)

#         labels = None
#         # # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
#         # labels[: -1] = IGNORE_INDEX
#         # if not self.predict_stop_token:
#         #     labels[-1] = IGNORE_INDEX

#         return_dict = dict(
#             pixel_values=pixel_values,
#             input_ids=input_ids,
#             labels=labels,
#             dataset_name=dataset_name,
#             actions=actions,
#             action_masks=action_mask,
#             current_state_mask=current_state_mask,
#             current_state=current_state,
#         )

#         return return_dict

if __name__ == '__main__':
    # Example usage - update paths to your data location
    video_root = '<VIDEO_ROOT>'
    egoexo4d_VLA_annotation = '<ANNOTATION_PATH>'
    dataset = EpisodicDatasetCore(video_root, egoexo4d_VLA_annotation)
    print(dataset)
