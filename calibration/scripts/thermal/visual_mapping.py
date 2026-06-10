#!/usr/bin/env python3
"""
可视化工具 - 使用标注文件将 RGB 图像中的点映射到 IR 图像，并对比可视化

使用方法:
    python visual_mapping.py <data_dir>

说明:
    - <data_dir>: 包含 rgb.png, rgb.yaml, thermal.png, thermal.yaml 的目录
    - 程序会在 data_dir 中查找最新的 calibration_* 目录中的单应矩阵文件
    - 程序会使用 homography 将 rgb.yaml 中的点映射到 IR 坐标系，并与 thermal.yaml 中的点一起可视化
    - 输出将保存到 <data_dir>/visualization_<timestamp>/
"""

import cv2
import numpy as np
import argparse
import os
import sys
import yaml
import time


# Add calibration root to path for src module imports
from pathlib import Path
calibration_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(calibration_root))

from src.thermal.coordinate_transformer import CoordinateTransformer, load_transformer_from_dir


def find_latest_calibration_dir(data_dir):
    """
    在 data_dir 中查找最新的 calibration_* 目录

    Args:
        data_dir: 数据目录路径

    Returns:
        最新的 calibration 目录路径，如果没有找到则返回 None
    """
    import glob

    calibration_dirs = glob.glob(os.path.join(data_dir, "calibration_*"))
    if not calibration_dirs:
        return None

    # 按修改时间排序，返回最新的
    calibration_dirs.sort(key=os.path.getmtime, reverse=True)
    return calibration_dirs[0]


def load_points_from_yaml(yaml_path):
    """
    从标注 YAML 文件中加载点坐标

    YAML 格式示例:
        image_path: /path/to/image.png
        points:
        - [x1, y1]
        - [x2, y2]
        ...
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"标注文件不存在: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    points = data.get("points", [])
    if not points:
        raise ValueError(f"标注文件中没有找到 points 字段或为空: {yaml_path}")

    return np.array(points, dtype=np.float32)


def draw_points(image, points, color=(0, 255, 0), radius=5, labels=None):
    """
    在图像上绘制点
    
    Args:
        image: 输入图像
        points: 点坐标列表
        color: 点的颜色 (B, G, R)
        radius: 点的半径
        labels: 可选的标签列表
    """
    for i, point in enumerate(points):
        x, y = int(point[0]), int(point[1])
        cv2.circle(image, (x, y), radius, color, -1)
        cv2.circle(image, (x, y), radius + 2, (255, 255, 255), 2)
        
        if labels is not None:
            cv2.putText(image, str(labels[i]), (x + 8, y - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


def draw_connections(image1, points1, image2, points2, color=(255, 0, 0), thickness=1):
    """
    在两个图像之间绘制连接线（用于并排显示）
    
    注意: 这个函数假设两个图像已经并排拼接在一起
    """
    if len(points1) != len(points2):
        return
    
    width1 = image1.shape[1]
    
    for p1, p2 in zip(points1, points2):
        x1, y1 = int(p1[0]), int(p1[1])
        x2, y2 = int(p2[0] + width1), int(p2[1])
        cv2.line(image1, (x1, y1), (x2, y2), color, thickness)


def visualize_mapping_with_annotations(
    rgb_image_path: str,
    ir_image_path: str,
    transformer: CoordinateTransformer,
    rgb_yaml_path: str,
    thermal_yaml_path: str,
    show_connections: bool = True,
    output_path: str | None = None,
):
    """
    使用标注文件可视化 RGB 点映射到 IR 点的效果，并与 IR 标注点对比。

    - 从 rgb_yaml_path 读取 RGB 图像中的标注点 P_rgb
    - 使用 transformer 将 P_rgb 映射到 IR 坐标系，得到 P_ir_pred
    - 从 thermal_yaml_path 读取 IR 图像中的标注点 P_ir_gt
    - 在 RGB 图像上画 P_rgb
    - 在 IR 图像上画 P_ir_gt（绿色）和 P_ir_pred（红色），并可选画连接线
    """
    # 加载图像
    rgb_image = cv2.imread(rgb_image_path)
    ir_image = cv2.imread(ir_image_path)
    
    if rgb_image is None:
        raise ValueError(f"无法加载RGB图像: {rgb_image_path}")
    if ir_image is None:
        raise ValueError(f"无法加载IR图像: {ir_image_path}")

    # 加载标注点
    rgb_points = load_points_from_yaml(rgb_yaml_path)  # (N, 2)
    ir_points_gt = load_points_from_yaml(thermal_yaml_path)  # (N, 2)

    if len(rgb_points) != len(ir_points_gt):
        print(
            f"警告: RGB 和 IR 标注点数量不一致 "
            f"({len(rgb_points)} vs {len(ir_points_gt)}), 将使用较少的一侧"
        )
        n = min(len(rgb_points), len(ir_points_gt))
        rgb_points = rgb_points[:n]
        ir_points_gt = ir_points_gt[:n]

    # 使用单应矩阵将 RGB 点映射到 IR 坐标系
    ir_points_pred = transformer.transform_rgb_to_ir(rgb_points.tolist())
    ir_points_pred = np.array(ir_points_pred, dtype=np.float32)

    # 创建副本用于绘制
    rgb_display = rgb_image.copy()
    ir_display = ir_image.copy()

    # 在 RGB 图像上绘制原始标注点
    draw_points(rgb_display, rgb_points, color=(0, 255, 0), radius=5)

    # 在 IR 图像上绘制 GT 点（绿色）和预测点（红色）
    draw_points(ir_display, ir_points_gt, color=(0, 255, 0), radius=5)   # GT
    draw_points(ir_display, ir_points_pred, color=(0, 0, 255), radius=4)  # Pred

    # 调整图像大小以便并排显示（保持宽高比）
    max_height = max(rgb_display.shape[0], ir_display.shape[0])

    # 缩放图像以匹配高度
    scale_rgb = max_height / rgb_display.shape[0]
    scale_ir = max_height / ir_display.shape[0]

    new_width_rgb = int(rgb_display.shape[1] * scale_rgb)
    new_width_ir = int(ir_display.shape[1] * scale_ir)

    rgb_resized = cv2.resize(rgb_display, (new_width_rgb, max_height))
    ir_resized = cv2.resize(ir_display, (new_width_ir, max_height))

    # 调整点坐标以匹配缩放后的图像
    rgb_points_scaled = [[p[0] * scale_rgb, p[1] * scale_rgb] for p in rgb_points]
    ir_points_gt_scaled = [[p[0] * scale_ir, p[1] * scale_ir] for p in ir_points_gt]
    ir_points_pred_scaled = [
        [p[0] * scale_ir, p[1] * scale_ir] for p in ir_points_pred
    ]

    # 并排拼接图像
    combined = np.hstack([rgb_resized, ir_resized])

    # 如果需要，绘制连接线（RGB 原始点 -> IR 预测点）
    if show_connections and len(rgb_points_scaled) <= 200:
        for p_rgb, p_ir_pred in zip(rgb_points_scaled, ir_points_pred_scaled):
            x1, y1 = int(p_rgb[0]), int(p_rgb[1])
            x2, y2 = int(p_ir_pred[0] + new_width_rgb), int(p_ir_pred[1])
            cv2.line(combined, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # 计算并打印平均投影误差（预测点 vs IR 标注点）
    diffs = ir_points_pred - ir_points_gt
    dists = np.linalg.norm(diffs, axis=1)
    mean_err = float(np.mean(dists))
    max_err = float(np.max(dists))
    print(f"投影误差: 平均 {mean_err:.2f} 像素, 最大 {max_err:.2f} 像素, 点数 {len(dists)}")

    # 添加标签
    cv2.putText(
        combined,
        "RGB Image (source points)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        combined,
        "IR Image (GT: green, Pred: red)",
        (new_width_rgb + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    # 显示或保存
    if output_path:
        cv2.imwrite(output_path, combined)
        print(f"可视化结果已保存到: {output_path}")
    else:
        cv2.imshow("RGB-IR Mapping Visualization", combined)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='使用标注文件可视化 RGB->IR 单应映射效果'
    )
    parser.add_argument('data_dir', type=str,
                       help='包含数据的目录 (必须包含 rgb.png, thermal.png, rgb.yaml, thermal.yaml)')
    parser.add_argument('--no_connections', action='store_true',
                       help='不显示连接线')

    args = parser.parse_args()

    # 生成带时间戳的输出目录和路径
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.data_dir, f"visualization_{timestamp}")
    output_path = os.path.join(output_dir, "mapping_result.png")
    os.makedirs(output_dir, exist_ok=True)

    # 构造并检查输入文件路径
    rgb_image_path = os.path.join(args.data_dir, "rgb.png")
    ir_image_path = os.path.join(args.data_dir, "thermal.png")
    rgb_yaml_path = os.path.join(args.data_dir, "rgb.yaml")
    thermal_yaml_path = os.path.join(args.data_dir, "thermal.yaml")

    required_files = {
        "RGB 图像": rgb_image_path,
        "IR 图像": ir_image_path,
        "RGB 标注": rgb_yaml_path,
        "IR 标注": thermal_yaml_path
    }

    for name, path in required_files.items():
        if not os.path.exists(path):
            print(f"错误: {name}不存在: {path}")
            return

    # 查找最新的 calibration 目录
    homography_dir = find_latest_calibration_dir(args.data_dir)
    if not homography_dir:
        print(f"错误: 在 {args.data_dir} 中未找到 calibration_* 目录")
        print("请先运行 homography_calculator.py 生成单应矩阵")
        return

    print(f"使用单应矩阵目录: {homography_dir}")
    
    try:
        # 加载坐标转换器
        print(f"从目录加载单应矩阵: {homography_dir}")
        transformer = load_transformer_from_dir(homography_dir)
        print("单应矩阵加载成功")

        # 可视化 RGB 标注点 -> IR 标注点 的映射
        print("\n根据标注文件生成可视化 ...")
        visualize_mapping_with_annotations(
            rgb_image_path=rgb_image_path,
            ir_image_path=ir_image_path,
            transformer=transformer,
            rgb_yaml_path=rgb_yaml_path,
            thermal_yaml_path=thermal_yaml_path,
            show_connections=not args.no_connections,
            output_path=output_path,
        )
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

