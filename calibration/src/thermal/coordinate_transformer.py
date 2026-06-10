#!/usr/bin/env python3
"""
坐标转换类 - 封装单应矩阵，提供RGB和IR之间的坐标转换功能

使用方法:
    from coordinate_transformer import CoordinateTransformer
    
    transformer = CoordinateTransformer(
        homography_rgb_to_ir_path='path/to/homography_rgb_to_ir.npy',
        homography_ir_to_rgb_path='path/to/homography_ir_to_rgb.npy'
    )
    
    # 转换单个点
    ir_point = transformer.transform_rgb_to_ir([100, 200])
    
    # 转换多个点
    rgb_points = [[100, 200], [150, 250], [200, 300]]
    ir_points = transformer.transform_rgb_to_ir(rgb_points)
"""

import numpy as np
import json
import os


class CoordinateTransformer:
    """坐标转换类，用于RGB和IR图像之间的坐标转换"""
    
    def __init__(self, homography_rgb_to_ir_path=None, homography_ir_to_rgb_path=None):
        """
        初始化坐标转换器
        
        Args:
            homography_rgb_to_ir_path: RGB->IR单应矩阵文件路径（.npy或.json）
            homography_ir_to_rgb_path: IR->RGB单应矩阵文件路径（.npy或.json）
        
        注意: 如果只提供一个路径，另一个方向的转换将使用逆矩阵
        """
        self.homography_rgb_to_ir = None
        self.homography_ir_to_rgb = None
        
        if homography_rgb_to_ir_path:
            self.homography_rgb_to_ir = self._load_homography(homography_rgb_to_ir_path)
        
        if homography_ir_to_rgb_path:
            self.homography_ir_to_rgb = self._load_homography(homography_ir_to_rgb_path)
        elif self.homography_rgb_to_ir is not None:
            # 如果没有提供IR->RGB的矩阵，使用逆矩阵
            self.homography_ir_to_rgb = np.linalg.inv(self.homography_rgb_to_ir)
        
        if self.homography_rgb_to_ir is None and self.homography_ir_to_rgb is not None:
            # 如果只提供了IR->RGB的矩阵，计算逆矩阵
            self.homography_rgb_to_ir = np.linalg.inv(self.homography_ir_to_rgb)
        
        if self.homography_rgb_to_ir is None:
            raise ValueError("必须至少提供一个单应矩阵文件路径")
    
    def _load_homography(self, file_path):
        """加载单应矩阵（支持.npy和.json格式）"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"单应矩阵文件不存在: {file_path}")
        
        if file_path.endswith('.npy'):
            return np.load(file_path)
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return np.array(data, dtype=np.float32)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}，请使用.npy或.json格式")
    
    def _transform_points(self, points, homography_matrix):
        """
        使用单应矩阵转换点坐标
        
        Args:
            points: 点坐标，可以是单个点 [x, y] 或批量点 [[x1, y1], [x2, y2], ...]
            homography_matrix: 3x3单应矩阵
        
        Returns:
            转换后的点坐标，格式与输入相同
        """
        # 转换为numpy数组
        points = np.array(points, dtype=np.float32)
        
        # 处理单个点的情况
        is_single_point = points.ndim == 1
        if is_single_point:
            points = points.reshape(1, -1)
        
        # 转换为齐次坐标
        n_points = len(points)
        homogeneous_points = np.ones((n_points, 3), dtype=np.float32)
        homogeneous_points[:, :2] = points
        
        # 应用单应矩阵变换
        transformed = (homography_matrix @ homogeneous_points.T).T
        
        # 归一化齐次坐标
        transformed_points = transformed[:, :2] / transformed[:, 2:3]
        
        # 如果输入是单个点，返回单个点
        if is_single_point:
            return transformed_points[0].tolist()
        else:
            return transformed_points.tolist()
    
    def transform_rgb_to_ir(self, points):
        """
        将RGB图像坐标转换为IR图像坐标
        
        Args:
            points: RGB图像中的点坐标，可以是单个点 [x, y] 或批量点 [[x1, y1], [x2, y2], ...]
        
        Returns:
            IR图像中对应的点坐标
        """
        if self.homography_rgb_to_ir is None:
            raise ValueError("RGB->IR单应矩阵未加载")
        
        return self._transform_points(points, self.homography_rgb_to_ir)
    
    def transform_ir_to_rgb(self, points):
        """
        将IR图像坐标转换为RGB图像坐标
        
        Args:
            points: IR图像中的点坐标，可以是单个点 [x, y] 或批量点 [[x1, y1], [x2, y2], ...]
        
        Returns:
            RGB图像中对应的点坐标
        """
        if self.homography_ir_to_rgb is None:
            raise ValueError("IR->RGB单应矩阵未加载")
        
        return self._transform_points(points, self.homography_ir_to_rgb)
    
    def get_homography_rgb_to_ir(self):
        """获取RGB->IR单应矩阵"""
        return self.homography_rgb_to_ir.copy() if self.homography_rgb_to_ir is not None else None
    
    def get_homography_ir_to_rgb(self):
        """获取IR->RGB单应矩阵"""
        return self.homography_ir_to_rgb.copy() if self.homography_ir_to_rgb is not None else None


def load_transformer_from_dir(homography_dir):
    """
    从目录加载单应矩阵并创建转换器
    
    Args:
        homography_dir: 包含单应矩阵文件的目录
    
    Returns:
        CoordinateTransformer实例
    """
    rgb_to_ir_npy = os.path.join(homography_dir, 'homography_rgb_to_ir.npy')
    ir_to_rgb_npy = os.path.join(homography_dir, 'homography_ir_to_rgb.npy')
    rgb_to_ir_json = os.path.join(homography_dir, 'homography_rgb_to_ir.json')
    ir_to_rgb_json = os.path.join(homography_dir, 'homography_ir_to_rgb.json')
    
    # 优先使用.npy格式
    rgb_to_ir_path = rgb_to_ir_npy if os.path.exists(rgb_to_ir_npy) else rgb_to_ir_json
    ir_to_rgb_path = ir_to_rgb_npy if os.path.exists(ir_to_rgb_npy) else ir_to_rgb_json
    
    if not os.path.exists(rgb_to_ir_path):
        raise FileNotFoundError(f"在目录 {homography_dir} 中找不到RGB->IR单应矩阵文件")
    
    return CoordinateTransformer(
        homography_rgb_to_ir_path=rgb_to_ir_path if os.path.exists(rgb_to_ir_path) else None,
        homography_ir_to_rgb_path=ir_to_rgb_path if os.path.exists(ir_to_rgb_path) else None
    )
