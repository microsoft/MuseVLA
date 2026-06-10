#!/usr/bin/env python3
"""
Multi-Image Annotation Tool - Annotate calibration points across multiple image pairs.

Usage:
    python annotate_multi.py <data_dir> --sensor acoustic -o <output.yaml>
    python annotate_multi.py <data_dir> --sensor rgb -o <output.yaml>

Keyboard shortcuts:
    - Left mouse button: Annotate point
    - Left/Right arrow: Switch between images
    - u or z: Undo last annotation on current image
    - r: Reset annotations on current image
    - s: Save all annotations and exit
    - q or ESC: Exit without saving
"""

import matplotlib
# Force interactive backend before importing pyplot
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
    import warnings
    warnings.warn("Unable to set interactive backend, mouse and keyboard interaction may not work")

import matplotlib.pyplot as plt
import numpy as np
import yaml
import argparse
import os
import glob
from pathlib import Path
from PIL import Image


class MultiImageAnnotationTool:
    def __init__(self, image_paths, output_path, sensor_type='acoustic'):
        """
        Initialize multi-image annotation tool.

        Args:
            image_paths: List of image file paths to annotate
            output_path: Output YAML file path
            sensor_type: Sensor type ('acoustic' or 'rgb')
        """
        self.image_paths = sorted(image_paths)
        self.output_path = output_path
        self.sensor_type = sensor_type
        self.current_idx = 0
        self.images = []
        self.annotations = []  # List of point lists, one per image

        # Load all images
        for path in self.image_paths:
            img = self._load_image(path)
            self.images.append(img)
            self.annotations.append([])

        if len(self.images) == 0:
            raise ValueError("No images found to annotate")

        # Check backend
        backend_name = matplotlib.get_backend()
        if backend_name == 'Agg':
            raise RuntimeError(
                "Currently using non-interactive backend (Agg).\n"
                "Please install an interactive backend:\n"
                "  - TkAgg: pip install tk\n"
                "  - Qt5Agg: pip install PyQt5"
            )

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        try:
            if hasattr(self.fig.canvas, 'manager') and hasattr(self.fig.canvas.manager, 'set_window_title'):
                self.fig.canvas.manager.set_window_title(f"Multi-Image Annotation Tool - {sensor_type}")
        except:
            pass

    def _load_image(self, image_path):
        """Load image from path."""
        try:
            pil_image = Image.open(image_path)
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            return np.array(pil_image)
        except Exception as e:
            raise ValueError(f"Failed to load image: {image_path}, error: {e}")

    def on_click(self, event):
        """Handle mouse click events."""
        if event.inaxes != self.ax:
            return

        if event.button == 1:  # Left button
            x, y = int(event.xdata), int(event.ydata)
            self.annotations[self.current_idx].append([x, y])
            self.update_display()
            print(f"[Image {self.current_idx + 1}/{len(self.images)}] Point {len(self.annotations[self.current_idx])}: ({x}, {y})")

    def on_key(self, event):
        """Handle keyboard events."""
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
            self.reset_current()
        elif event.key == 'left':
            self.prev_image()
        elif event.key == 'right':
            self.next_image()

    def prev_image(self):
        """Switch to previous image."""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.update_display()
            print(f"Switched to image {self.current_idx + 1}/{len(self.images)}")

    def next_image(self):
        """Switch to next image."""
        if self.current_idx < len(self.images) - 1:
            self.current_idx += 1
            self.update_display()
            print(f"Switched to image {self.current_idx + 1}/{len(self.images)}")

    def update_display(self):
        """Update display with current image and annotations."""
        self.ax.clear()
        self.ax.imshow(self.images[self.current_idx])

        # Title with navigation info
        filename = os.path.basename(self.image_paths[self.current_idx])
        n_points = len(self.annotations[self.current_idx])
        total_points = sum(len(pts) for pts in self.annotations)
        title = (
            f"[{self.current_idx + 1}/{len(self.images)}] {filename} | "
            f"Points: {n_points} (Total: {total_points})\n"
            f"← → to switch | Click to annotate | 's' save | 'q' quit"
        )
        self.ax.set_title(title, fontsize=12)
        self.ax.axis('off')

        # Draw annotations for current image
        for i, point in enumerate(self.annotations[self.current_idx]):
            x, y = point
            circle = plt.Circle((x, y), 5, color='green', fill=True, zorder=5)
            self.ax.add_patch(circle)
            self.ax.text(x + 10, y - 10, str(i + 1),
                        color='green', fontsize=12, weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
                        zorder=6)

        self.fig.canvas.draw()

    def undo(self):
        """Undo last annotation on current image."""
        if len(self.annotations[self.current_idx]) > 0:
            removed = self.annotations[self.current_idx].pop()
            self.update_display()
            print(f"Undone point: ({removed[0]}, {removed[1]})")
        else:
            print("No points to undo on current image")

    def reset_current(self):
        """Reset annotations on current image."""
        self.annotations[self.current_idx] = []
        self.update_display()
        print(f"Reset annotations on image {self.current_idx + 1}")

    def sort_points(self, points, image_height):
        """Sort points from left to right, top to bottom."""
        if len(points) == 0:
            return points

        points_array = np.array(points)
        y_threshold = image_height * 0.05

        sorted_indices = np.argsort(points_array[:, 1])
        sorted_points = points_array[sorted_indices]

        rows = []
        current_row = [sorted_points[0]]

        for i in range(1, len(sorted_points)):
            if abs(sorted_points[i][1] - current_row[-1][1]) < y_threshold:
                current_row.append(sorted_points[i])
            else:
                current_row = np.array(current_row)
                current_row = current_row[np.argsort(current_row[:, 0])]
                rows.append(current_row.tolist())
                current_row = [sorted_points[i]]

        if len(current_row) > 0:
            current_row = np.array(current_row)
            current_row = current_row[np.argsort(current_row[:, 0])]
            rows.append(current_row.tolist())

        return [point for row in rows for point in row]

    def save(self):
        """Save all annotations to YAML file."""
        total_points = sum(len(pts) for pts in self.annotations)
        if total_points == 0:
            print("Warning: No annotation points, not saving")
            return False

        # Prepare data
        images_data = []
        for i, (path, points) in enumerate(zip(self.image_paths, self.annotations)):
            if len(points) > 0:
                sorted_pts = self.sort_points(points, self.images[i].shape[0])
                images_data.append({
                    'path': os.path.basename(path),
                    'points': sorted_pts
                })

        data = {
            'sensor_type': self.sensor_type,
            'images': images_data,
            'total_points': total_points
        }

        # Save as YAML
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(self.output_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        print(f"Saved {total_points} annotation points across {len(images_data)} images to: {self.output_path}")
        return True

    def run(self):
        """Run annotation tool."""
        print("\n=== Multi-Image Annotation Tool ===")
        print(f"Sensor type: {self.sensor_type}")
        print(f"Images to annotate: {len(self.images)}")
        print("-" * 40)
        print("Left mouse button: Annotate point")
        print("← / →: Switch between images")
        print("u or z: Undo last annotation")
        print("r: Reset current image annotations")
        print("s: Save and exit")
        print("q or ESC: Exit without saving")
        print("=" * 40 + "\n")

        self.update_display()
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Multi-image annotation tool for acoustic calibration'
    )
    parser.add_argument('data_dir', type=str,
                       help='Directory containing calibration images (e.g., 000_acoustic.png, 000_rgb.png)')
    parser.add_argument('--sensor', type=str, choices=['acoustic', 'rgb'], required=True,
                       help='Which sensor type to annotate (acoustic or rgb)')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output YAML file path. Default: <data_dir>/<sensor>.yaml')

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"Error: Directory does not exist: {args.data_dir}")
        return

    # Find matching images
    if args.sensor == 'acoustic':
        pattern = os.path.join(args.data_dir, '*_acoustic.png')
    else:
        pattern = os.path.join(args.data_dir, '*_rgb.png')

    image_paths = sorted(glob.glob(pattern))

    if len(image_paths) == 0:
        print(f"Error: No {args.sensor} images found with pattern: {pattern}")
        return

    print(f"Found {len(image_paths)} {args.sensor} images")

    # Set output path
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(args.data_dir, f'{args.sensor}.yaml')
        print(f"No output path specified. Using default: {output_path}")

    try:
        tool = MultiImageAnnotationTool(image_paths, output_path, args.sensor)
        tool.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
