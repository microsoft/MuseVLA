#!/usr/bin/env python3
"""
单应矩阵计算模块 - 根据RGB和IR图像的标注点计算单应矩阵

使用方法:
    python homography_calculator.py <data_dir>

说明:
    - <data_dir>: 包含 rgb.yaml 和 thermal.yaml 标注文件的目录
    - 输出将保存到 <data_dir>/calibration_<timestamp>/
"""

import cv2
import numpy as np
import yaml
import json
import argparse
import os
import time
from pathlib import Path


def load_annotations(yaml_path):
    """从YAML文件加载标注点"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    points = data.get('points', [])
    if len(points) < 4:
        raise ValueError(f"标注点数量不足（至少需要4个点，当前有{len(points)}个）")
    
    return np.array(points, dtype=np.float32)


def calculate_homography(src_points, dst_points):
    """
    计算单应矩阵
    
    Args:
        src_points: 源图像点坐标 (N, 2)
        dst_points: 目标图像点坐标 (N, 2)
    
    Returns:
        3x3单应矩阵
    """
    if len(src_points) != len(dst_points):
        raise ValueError("源点和目标点数量必须相同")
    
    if len(src_points) < 4:
        raise ValueError("至少需要4个对应点来计算单应矩阵")
    
    # 使用RANSAC方法计算单应矩阵，提高鲁棒性
    homography, mask = cv2.findHomography(
        src_points, 
        dst_points, 
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0
    )
    
    if homography is None:
        raise ValueError("无法计算单应矩阵，请检查标注点是否正确对应")
    
    # 统计内点数量
    inliers = np.sum(mask)
    print(f"单应矩阵计算完成: {inliers}/{len(src_points)} 个内点")
    
    return homography


def save_homography(homography, output_dir, prefix):
    """
    保存单应矩阵为JSON和.npy格式
    
    Args:
        homography: 3x3单应矩阵
        output_dir: 输出目录
        prefix: 文件名前缀（如 'rgb_to_ir'）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为JSON格式
    json_path = os.path.join(output_dir, f'homography_{prefix}.json')
    homography_list = homography.tolist()
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(homography_list, f, indent=2)
    print(f"已保存JSON格式: {json_path}")
    
    # 保存为.npy格式
    npy_path = os.path.join(output_dir, f'homography_{prefix}.npy')
    np.save(npy_path, homography)
    print(f"已保存NumPy格式: {npy_path}")
    
    return json_path, npy_path


def main():
    parser = argparse.ArgumentParser(
        description='计算RGB和IR图像之间的单应矩阵'
    )
    parser.add_argument('data_dir', type=str,
                       help='包含 rgb.yaml 和 thermal.yaml 的数据目录')

    args = parser.parse_args()

    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"错误: 数据目录不存在: {args.data_dir}")
        return

    # 构造标注文件路径
    rgb_annotations = os.path.join(args.data_dir, "rgb.yaml")
    ir_annotations = os.path.join(args.data_dir, "thermal.yaml")

    # 检查输入文件
    if not os.path.exists(rgb_annotations):
        print(f"错误: RGB标注文件不存在: {rgb_annotations}")
        return

    if not os.path.exists(ir_annotations):
        print(f"错误: IR标注文件不存在: {ir_annotations}")
        return

    # 生成带时间戳的输出目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.data_dir, f"calibration_{timestamp}")
    
    try:
        # 加载标注点
        print(f"加载RGB标注: {rgb_annotations}")
        rgb_points = load_annotations(rgb_annotations)
        print(f"RGB标注点数量: {len(rgb_points)}")

        print(f"加载IR标注: {ir_annotations}")
        ir_points = load_annotations(ir_annotations)
        print(f"IR标注点数量: {len(ir_points)}")

        if len(rgb_points) != len(ir_points):
            print(f"警告: RGB和IR标注点数量不一致 ({len(rgb_points)} vs {len(ir_points)})")
            print("将使用较少的点数")
            min_points = min(len(rgb_points), len(ir_points))
            rgb_points = rgb_points[:min_points]
            ir_points = ir_points[:min_points]

        # 计算RGB->IR的单应矩阵
        print("\n计算 RGB -> IR 单应矩阵...")
        homography_rgb_to_ir = calculate_homography(rgb_points, ir_points)
        save_homography(homography_rgb_to_ir, output_dir, 'rgb_to_ir')

        # 计算IR->RGB的单应矩阵（使用逆矩阵）
        print("\n计算 IR -> RGB 单应矩阵...")
        homography_ir_to_rgb = np.linalg.inv(homography_rgb_to_ir)
        save_homography(homography_ir_to_rgb, output_dir, 'ir_to_rgb')

        # 验证：计算IR->RGB的单应矩阵（从另一个方向）
        print("\n验证: 从IR->RGB方向计算单应矩阵...")
        homography_ir_to_rgb_verify = calculate_homography(ir_points, rgb_points)
        save_homography(homography_ir_to_rgb_verify, output_dir, 'ir_to_rgb_verify')

        print("\n单应矩阵计算完成！")
        print(f"输出目录: {output_dir}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
