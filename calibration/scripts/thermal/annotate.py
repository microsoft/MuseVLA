#!/usr/bin/env python3
"""
Annotation Tool - Annotate thermal source center points on images (using Matplotlib backend)

Usage:
    python annotate.py <image_path> -o <output_yaml>

Keyboard shortcuts:
    - Left mouse button: Annotate point
    - u or z: Undo last annotation
    - r: Reset all annotations
    - s: Save and exit
    - q or ESC: Exit without saving
"""

import matplotlib
# 强制使用交互式后端（在导入 pyplot 之前）
# 尝试多个交互式后端，按优先级顺序
backends_to_try = ['TkAgg', 'Qt5Agg', 'Qt4Agg']
backend_set = False
for backend in backends_to_try:
    try:
        matplotlib.use(backend)
        backend_set = True
        break
    except:
        continue

if not backend_set:
    # If all interactive backends are unavailable, issue a warning
    import warnings
    warnings.warn("Unable to set interactive backend, mouse and keyboard interaction may not work")

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import yaml
import argparse
import os
import time
from pathlib import Path
from PIL import Image


class AnnotationTool:
    def __init__(self, image_path, output_path):
        self.image_path = image_path
        self.output_path = output_path
        self.points = []
        
        # 加载图像（优先使用 PIL，兼容性更好）
        try:
            # 尝试使用 PIL 加载（支持更多格式）
            pil_image = Image.open(image_path)
            # 转换为 RGB（如果是 RGBA 或其他格式）
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            # 转换为 numpy 数组
            self.image = np.array(pil_image)
        except Exception as e:
            # 如果 PIL 失败，尝试 matplotlib
            try:
                self.image = mpimg.imread(image_path)
            except Exception as e2:
                raise ValueError(f"Failed to load image: {image_path}, PIL error: {e}, Matplotlib error: {e2}")
        
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Check if backend supports interaction
        backend_name = matplotlib.get_backend()
        if backend_name == 'Agg':
            raise RuntimeError(
                "Currently using non-interactive backend (Agg), mouse and keyboard interaction unavailable.\n"
                "Please install an interactive backend:\n"
                "  - TkAgg: pip install tk\n"
                "  - Qt5Agg: pip install PyQt5\n"
                "Or force set backend in code."
            )
        
        # Create figure and axes
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(self.image)
        self.ax.set_title("Annotation Tool - Click to annotate, press 's' to save, 'q' to quit", fontsize=14)
        self.ax.axis('off')  # Hide axes
        
        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Set window title (if backend supports it)
        try:
            if hasattr(self.fig.canvas, 'set_window_title'):
                self.fig.canvas.set_window_title("Annotation Tool")
            elif hasattr(self.fig.canvas, 'manager') and hasattr(self.fig.canvas.manager, 'set_window_title'):
                self.fig.canvas.manager.set_window_title("Annotation Tool")
            else:
                # Use figure suptitle as alternative
                self.fig.suptitle("Annotation Tool", fontsize=16)
        except:
            # Ignore error if setting title fails
            pass
        
    def on_click(self, event):
        """Handle mouse click events"""
        if event.inaxes != self.ax:
            return
        
        if event.button == 1:  # Left button
            x, y = int(event.xdata), int(event.ydata)
            self.points.append([x, y])
            self.update_display()
            print(f"Point {len(self.points)}: ({x}, {y})")
    
    def on_key(self, event):
        """Handle keyboard events"""
        if event.key == 'q' or event.key == 'escape':
            print("Exiting (not saved)")
            plt.close(self.fig)
        elif event.key == 's':
            if self.save():
                print("Saved successfully, exiting")
                plt.close(self.fig)
        elif event.key == 'u' or event.key == 'z':
            self.undo()
        elif event.key == 'r':
            self.reset()
    
    def update_display(self):
        """Update display image, draw all annotated points"""
        self.ax.clear()
        self.ax.imshow(self.image)
        self.ax.set_title("Annotation Tool - Click to annotate, press 's' to save, 'q' to quit", fontsize=14)
        self.ax.axis('off')
        
        # Draw annotated points
        for i, point in enumerate(self.points):
            x, y = point
            # Draw point (green circle)
            circle = plt.Circle((x, y), 5, color='green', fill=True, zorder=5)
            self.ax.add_patch(circle)
            # Draw index number
            self.ax.text(x + 10, y - 10, str(i + 1), 
                        color='green', fontsize=12, weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                        zorder=6)
        
        self.fig.canvas.draw()
    
    def sort_points(self):
        """Sort points from left to right, top to bottom"""
        if len(self.points) == 0:
            return
        
        # Convert to numpy array for easier processing
        points_array = np.array(self.points)
        
        # Calculate y-coordinate difference threshold (5% of image height as row spacing threshold)
        y_threshold = self.image.shape[0] * 0.05
        
        # Sort by y-coordinate
        sorted_indices = np.argsort(points_array[:, 1])
        sorted_points = points_array[sorted_indices]
        
        # Group: points with similar y-coordinates into the same row
        rows = []
        current_row = [sorted_points[0]]
        
        for i in range(1, len(sorted_points)):
            if abs(sorted_points[i][1] - current_row[-1][1]) < y_threshold:
                current_row.append(sorted_points[i])
            else:
                # Current row ends, sort by x-coordinate and add to rows
                current_row = np.array(current_row)
                current_row = current_row[np.argsort(current_row[:, 0])]
                rows.append(current_row.tolist())
                current_row = [sorted_points[i]]
        
        # Process last row
        if len(current_row) > 0:
            current_row = np.array(current_row)
            current_row = current_row[np.argsort(current_row[:, 0])]
            rows.append(current_row.tolist())
        
        # Flatten all rows
        self.points = [point for row in rows for point in row]
        
        print(f"Sorted {len(self.points)} points")
    
    def undo(self):
        """Undo last annotation"""
        if len(self.points) > 0:
            removed = self.points.pop()
            self.update_display()
            print(f"Undone point: ({removed[0]}, {removed[1]})")
        else:
            print("No points to undo")
    
    def reset(self):
        """Reset all annotations"""
        self.points = []
        self.update_display()
        print("Reset all annotations")
    
    def save(self):
        """Save annotations to YAML file"""
        if len(self.points) == 0:
            print("Warning: No annotation points, not saving")
            return False
        
        # Sort points
        self.sort_points()
        
        # Prepare data
        data = {
            'image_path': os.path.abspath(self.image_path),
            'points': self.points
        }
        
        # Save as YAML
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(self.output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        
        print(f"Saved {len(self.points)} annotation points to: {self.output_path}")
        return True
    
    def run(self):
        """Run annotation tool"""
        print("\n=== Annotation Tool Instructions ===")
        print("Left mouse button: Annotate point")
        print("u or z: Undo last annotation")
        print("r: Reset all annotations")
        print("s: Save and exit")
        print("q or ESC: Exit without saving")
        print("===================================\n")
        print("Tip: Click on the image window to give it focus, then you can use keyboard shortcuts")
        
        # Initial display
        self.update_display()
        
        # Show window (blocks until closed)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Image annotation tool - Annotate thermal source center points')
    parser.add_argument('image_path', type=str, help='Input image path')
    
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output YAML file path. If not provided, saves to the same directory as input image with .yaml extension')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file does not exist: {args.image_path}")
        return

    output_path = args.output
    if output_path is None:
        # 默认保存在原图片同目录下，文件名相同，扩展名改为 .yaml
        image_dir = os.path.dirname(os.path.abspath(args.image_path))
        image_name = os.path.splitext(os.path.basename(args.image_path))[0]
        output_path = os.path.join(image_dir, f"{image_name}.yaml")
        print(f"No output path specified. Using default: {output_path}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        tool = AnnotationTool(args.image_path, output_path)
        tool.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
