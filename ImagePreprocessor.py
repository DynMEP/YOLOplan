# =============================================================================
# YOLOplan: Automated Symbol Detection for Technical Drawings
# =============================================================================
# Purpose: Complete YOLO11 implementation for automated symbol detection and 
#          counting in technical drawings (PDF, images, CAD) for MEP 
#          (Mechanical, Electrical, Plumbing) projects with netlist generation.
# Version: 1.1.0 (Fixed & Enhanced Edition)
# Author: Alfonso Davila - Electrical Engineer, Revit MEP Dynamo BIM Expert
# Contact: davila.alfonso@gmail.com - www.linkedin.com/in/alfonso-davila-3a121087
# Repository: https://github.com/DynMEP/YOLOplan
# License: MIT License (see LICENSE file in repository)
# YOLO11: AGPL-3.0 (Ultralytics) - Commercial license available from Ultralytics
# Created: October 2025
# Last Updated: October 8, 2025
# Compatibility: Python 3.8+, CUDA 11.8+ (optional), Poppler (for PDF support)
# Quick Start:
#   Detection:  python YOLOplanDetector.py --model yolo11s.pt --source image.jpg
#   Training:   python YOLOplanTrainer.py train --data data.yaml --model s
#   Setup:      python YOLOplanTrainer.py setup --path datasets/my_data --classes c1 c2
# =============================================================================

import os
import cv2
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
   
    @staticmethod
    def estimate_noise(image: np.ndarray) -> float:
        try:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            return float(np.var(laplacian))
        except Exception as e:
            logger.warning(f"Error estimating noise: {e}")
            return 0.0
    
    @staticmethod
    def analyze_contrast(image: np.ndarray) -> float:
        try:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            hist = hist / hist.sum()
            return float(np.std(hist))
        except Exception as e:
            logger.warning(f"Error analyzing contrast: {e}")
            return 0.0

    def enhance_drawing(self, image_path: str, output_path: str = None,
                       sharpen: bool = True, denoise: bool = True,
                       contrast: bool = True) -> np.ndarray:
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Adaptive parameters based on image characteristics
            noise_level = self.estimate_noise(gray)
            contrast_score = self.analyze_contrast(gray)
            
            # Adaptive denoising
            if denoise and noise_level > 100:
                denoise_strength = 10 if noise_level > 1000 else 5
                gray = cv2.fastNlMeansDenoising(gray, None, denoise_strength, 7, 21)
            
            # Adaptive contrast enhancement
            if contrast:
                clip_limit = 2.0 if contrast_score < 0.02 else 1.5
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                gray = clahe.apply(gray)
            
            # Sharpening
            if sharpen:
                kernel = np.array([[-1, -1, -1],
                                 [-1, 9, -1],
                                 [-1, -1, -1]])
                gray = cv2.filter2D(gray, -1, kernel)
            
            enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            if output_path:
                cv2.imwrite(output_path, enhanced)
                logger.info(f"Saved enhanced image to {output_path}")
            
            return enhanced
        
        except Exception as e:
            logger.error(f"Error enhancing image {image_path}: {e}")
            raise
    
    def resize_maintain_aspect(self, image_path: str, target_size: int = 640,
                              output_path: str = None) -> np.ndarray:
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            h, w = img.shape[:2]
            
            if h > w:
                new_h = target_size
                new_w = int(w * (target_size / h))
            else:
                new_w = target_size
                new_h = int(h * (target_size / w))
            
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            if output_path:
                cv2.imwrite(output_path, resized)
                logger.info(f"Saved resized image to {output_path}")
            
            return resized
        
        except Exception as e:
            logger.error(f"Error resizing image {image_path}: {e}")
            raise
    
    def remove_background(self, image_path: str, output_path: str = None,
                         threshold: int = 240) -> np.ndarray:
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
            result = cv2.bitwise_and(img, img, mask=mask)
            
            if output_path:
                cv2.imwrite(output_path, result)
                logger.info(f"Saved background-removed image to {output_path}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error removing background from {image_path}: {e}")
            raise
    
    def generate_synthetic_schematic(self, class_names: List[str], output_dir: str, 
                                   num_images: int = 100, img_size: int = 640) -> None:
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Generating {num_images} synthetic schematics...")
            
            for i in tqdm(range(num_images), desc="Generating synthetics"):
                img = Image.new('RGB', (img_size, img_size), color='white')
                draw = ImageDraw.Draw(img)
                labels = []
                
                # Random number of symbols per image
                num_symbols = random.randint(5, 20)
                
                for _ in range(num_symbols):
                    cls_idx = random.randint(0, len(class_names) - 1)
                    cls = class_names[cls_idx]
                    
                    # Random position and size
                    x = random.randint(50, img_size - 100)
                    y = random.randint(50, img_size - 100)
                    size = random.randint(20, 60)
                    
                    # Draw random shapes for different symbol types
                    shape_type = random.choice(['rectangle', 'circle', 'triangle'])
                    
                    if shape_type == 'rectangle':
                        draw.rectangle((x, y, x + size, y + size), 
                                     outline='black', width=2)
                    elif shape_type == 'circle':
                        draw.ellipse((x, y, x + size, y + size), 
                                   outline='black', width=2)
                    else:  # triangle
                        draw.polygon([(x, y + size), (x + size//2, y), (x + size, y + size)],
                                   outline='black', width=2)
                    
                    # Add connecting lines occasionally
                    if random.random() < 0.3:
                        end_x = random.randint(x, min(x + 100, img_size - 10))
                        end_y = random.randint(y, min(y + 100, img_size - 10))
                        draw.line((x + size//2, y + size//2, end_x, end_y), 
                                fill='black', width=1)
                    
                    # Calculate YOLO format coordinates
                    x_center = (x + size/2) / img_size
                    y_center = (y + size/2) / img_size
                    norm_width = size / img_size
                    norm_height = size / img_size
                    
                    labels.append(f"{cls_idx} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")
                
                # Save image and label
                img_path = os.path.join(output_dir, f"synth_{i:04d}.jpg")
                label_path = os.path.join(output_dir, f"synth_{i:04d}.txt")
                
                img.save(img_path)
                with open(label_path, 'w') as f:
                    f.write('\n'.join(labels))
            
            logger.info(f"Generated {num_images} synthetic images in {output_dir}")
        
        except Exception as e:
            logger.error(f"Error generating synthetic schematics: {e}")
            raise
    
    def batch_preprocess_parallel(self, input_dir: str, output_dir: str,
                                enhance: bool = True, resize: int = None,
                                max_workers: int = None):
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Find all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_paths = []
            for ext in image_extensions:
                image_paths.extend(list(Path(input_dir).glob(f'*{ext}')))
                image_paths.extend(list(Path(input_dir).glob(f'*{ext.upper()}')))
            
            if not image_paths:
                logger.warning(f"No images found in {input_dir}")
                return
            
            logger.info(f"Processing {len(image_paths)} images in parallel...")
            
            def process_image(img_path):
                try:
                    output_path = os.path.join(output_dir, img_path.name)
                    
                    if enhance:
                        img = self.enhance_drawing(str(img_path))
                    else:
                        img = cv2.imread(str(img_path))
                        if img is None:
                            raise ValueError(f"Could not read image: {img_path}")
                    
                    if resize:
                        h, w = img.shape[:2]
                        if h > w:
                            new_h = resize
                            new_w = int(w * (resize / h))
                        else:
                            new_w = resize
                            new_h = int(h * (resize / w))
                        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    cv2.imwrite(output_path, img)
                    return True, img_path.name
                except Exception as e:
                    return False, f"{img_path.name}: {str(e)}"
            
            # Process in parallel with progress bar
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_image, img_path) for img_path in image_paths]
                
                success_count = 0
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                    success, message = future.result()
                    if success:
                        success_count += 1
                    else:
                        logger.warning(f"Failed: {message}")
            
            logger.info(f"Processing complete. {success_count}/{len(image_paths)} successful. Output: {output_dir}")
        
        except Exception as e:
            logger.error(f"Error in batch preprocessing: {e}")
            raise


class DatasetSplitter:
    
    @staticmethod
    def split_dataset(images_dir: str, labels_dir: str,
                     output_dir: str, train_ratio: float = 0.7,
                     val_ratio: float = 0.2, test_ratio: float = 0.1,
                     seed: int = 42):
        try:
            if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
                raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
            
            # Find all image files
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend([f.name for f in Path(images_dir).glob(ext)])
                image_files.extend([f.name for f in Path(images_dir).glob(ext.upper())])
            
            # Remove duplicates
            image_files = list(set(image_files))
            
            if not image_files:
                raise ValueError(f"No images found in {images_dir}")
            
            logger.info(f"Found {len(image_files)} images")
            
            # Split dataset
            train_files, temp_files = train_test_split(
                image_files, train_size=train_ratio, random_state=seed
            )
            val_size = val_ratio / (val_ratio + test_ratio)
            val_files, test_files = train_test_split(
                temp_files, train_size=val_size, random_state=seed
            )
            
            logger.info(f"Split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
            
            splits = {
                'train': train_files,
                'val': val_files,
                'test': test_files
            }
            
            # Copy files to respective directories
            for split_name, file_list in splits.items():
                img_dir = Path(output_dir) / 'images' / split_name
                lbl_dir = Path(output_dir) / 'labels' / split_name
                img_dir.mkdir(parents=True, exist_ok=True)
                lbl_dir.mkdir(parents=True, exist_ok=True)
                
                for filename in tqdm(file_list, desc=f"Copying {split_name}"):
                    # Copy image
                    src_img = Path(images_dir) / filename
                    dst_img = img_dir / filename
                    if src_img.exists():
                        shutil.copy2(src_img, dst_img)
                    else:
                        logger.warning(f"Image not found: {src_img}")
                        continue
                    
                    # Copy label
                    label_name = Path(filename).stem + '.txt'
                    src_lbl = Path(labels_dir) / label_name
                    dst_lbl = lbl_dir / label_name
                    if src_lbl.exists():
                        shutil.copy2(src_lbl, dst_lbl)
                    else:
                        logger.warning(f"Label not found for {filename}")
            
            logger.info(f"Dataset split complete. Output: {output_dir}")
        
        except Exception as e:
            logger.error(f"Error splitting dataset: {e}")
            raise


class DatasetAnalyzer:
    
    @staticmethod
    def analyze_dataset(labels_dir: str, class_names: List[str],
                       output_dir: str = 'analysis'):
        try:
            os.makedirs(output_dir, exist_ok=True)
            label_files = list(Path(labels_dir).glob('*.txt'))
            
            if not label_files:
                logger.warning(f"No label files found in {labels_dir}")
                return
            
            logger.info(f"Analyzing {len(label_files)} label files...")
            
            class_counts = {i: 0 for i in range(len(class_names))}
            bbox_areas = []
            bbox_widths = []
            bbox_heights = []
            bbox_aspect_ratios = []
            total_objects = 0
            images_with_objects = 0
            objects_per_image = []
            
            for label_file in tqdm(label_files, desc="Analyzing labels"):
                try:
                    with open(label_file, 'r') as f:
                        lines = f.readlines()
                        
                    num_objects = 0
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])
                            
                            if cls < len(class_names):
                                class_counts[cls] += 1
                                bbox_areas.append(width * height)
                                bbox_widths.append(width)
                                bbox_heights.append(height)
                                bbox_aspect_ratios.append(width / height if height > 0 else 1.0)
                                total_objects += 1
                                num_objects += 1
                    
                    if num_objects > 0:
                        images_with_objects += 1
                        objects_per_image.append(num_objects)
                
                except Exception as e:
                    logger.warning(f"Error reading {label_file}: {e}")
                    continue
            
            # Print statistics
            print("\n" + "="*60)
            print("DATASET STATISTICS")
            print("="*60)
            print(f"Total images: {len(label_files)}")
            print(f"Images with objects: {images_with_objects}")
            print(f"Total objects: {total_objects}")
            if images_with_objects > 0:
                print(f"Average objects per image: {np.mean(objects_per_image):.2f}")
                print(f"Min/Max objects per image: {min(objects_per_image)}/{max(objects_per_image)}")
            
            print("\nClass distribution:")
            for cls_id in range(len(class_names)):
                count = class_counts.get(cls_id, 0)
                percentage = (count / total_objects * 100) if total_objects > 0 else 0
                print(f"  {class_names[cls_id]}: {count} ({percentage:.1f}%)")
            
            if bbox_areas:
                print(f"\nBounding box statistics:")
                print(f"  Average area: {np.mean(bbox_areas):.4f}")
                print(f"  Average width: {np.mean(bbox_widths):.4f}")
                print(f"  Average height: {np.mean(bbox_heights):.4f}")
                print(f"  Average aspect ratio: {np.mean(bbox_aspect_ratios):.2f}")
            
            # Generate plots
            DatasetAnalyzer._plot_class_distribution(class_counts, class_names, output_dir)
            DatasetAnalyzer._plot_bbox_statistics(bbox_widths, bbox_heights, bbox_areas, 
                                                  bbox_aspect_ratios, output_dir)
            
            # Save statistics to file
            stats_file = os.path.join(output_dir, 'dataset_stats.txt')
            with open(stats_file, 'w') as f:
                f.write("DATASET STATISTICS\n")
                f.write("="*60 + "\n")
                f.write(f"Total images: {len(label_files)}\n")
                f.write(f"Images with objects: {images_with_objects}\n")
                f.write(f"Total objects: {total_objects}\n")
                if images_with_objects > 0:
                    f.write(f"Average objects per image: {np.mean(objects_per_image):.2f}\n")
                f.write("\nClass distribution:\n")
                for cls_id in range(len(class_names)):
                    count = class_counts.get(cls_id, 0)
                    percentage = (count / total_objects * 100) if total_objects > 0 else 0
                    f.write(f"  {class_names[cls_id]}: {count} ({percentage:.1f}%)\n")
            
            logger.info(f"Analysis complete. Results saved to: {output_dir}")
        
        except Exception as e:
            logger.error(f"Error analyzing dataset: {e}")
            raise
    
    @staticmethod
    def _plot_class_distribution(class_counts: Dict, class_names: List[str], output_dir: str):
        try:
            plt.figure(figsize=(12, 6))
            classes = [class_names[i] for i in sorted(class_counts.keys()) if i < len(class_names)]
            counts = [class_counts[i] for i in sorted(class_counts.keys()) if i < len(class_names)]
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
            bars = plt.bar(range(len(classes)), counts, color=colors, alpha=0.8, edgecolor='black')
            
            plt.xlabel('Class', fontsize=12, fontweight='bold')
            plt.ylabel('Count', fontsize=12, fontweight='bold')
            plt.title('Class Distribution', fontsize=14, fontweight='bold')
            plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3, linestyle='--')
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, 'class_distribution.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved class distribution plot to {output_path}")
        
        except Exception as e:
            logger.error(f"Error plotting class distribution: {e}")
    
    @staticmethod
    def _plot_bbox_statistics(widths: List, heights: List, areas: List, 
                             aspect_ratios: List, output_dir: str):
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Width distribution
            axes[0, 0].hist(widths, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
            axes[0, 0].set_xlabel('Width (normalized)', fontsize=10, fontweight='bold')
            axes[0, 0].set_ylabel('Frequency', fontsize=10, fontweight='bold')
            axes[0, 0].set_title('Bounding Box Width Distribution', fontsize=12, fontweight='bold')
            axes[0, 0].grid(alpha=0.3, linestyle='--')
            axes[0, 0].axvline(np.mean(widths), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(widths):.3f}')
            axes[0, 0].legend()
            
            # Height distribution
            axes[0, 1].hist(heights, bins=50, color='lightcoral', alpha=0.7, edgecolor='black')
            axes[0, 1].set_xlabel('Height (normalized)', fontsize=10, fontweight='bold')
            axes[0, 1].set_ylabel('Frequency', fontsize=10, fontweight='bold')
            axes[0, 1].set_title('Bounding Box Height Distribution', fontsize=12, fontweight='bold')
            axes[0, 1].grid(alpha=0.3, linestyle='--')
            axes[0, 1].axvline(np.mean(heights), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(heights):.3f}')
            axes[0, 1].legend()
            
            # Area distribution
            axes[1, 0].hist(areas, bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
            axes[1, 0].set_xlabel('Area (normalized)', fontsize=10, fontweight='bold')
            axes[1, 0].set_ylabel('Frequency', fontsize=10, fontweight='bold')
            axes[1, 0].set_title('Bounding Box Area Distribution', fontsize=12, fontweight='bold')
            axes[1, 0].grid(alpha=0.3, linestyle='--')
            axes[1, 0].axvline(np.mean(areas), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(areas):.4f}')
            axes[1, 0].legend()
            
            # Width vs Height scatter
            axes[1, 1].scatter(widths, heights, alpha=0.5, s=10, color='purple', edgecolors='black')
            axes[1, 1].set_xlabel('Width (normalized)', fontsize=10, fontweight='bold')
            axes[1, 1].set_ylabel('Height (normalized)', fontsize=10, fontweight='bold')
            axes[1, 1].set_title('Width vs Height', fontsize=12, fontweight='bold')
            axes[1, 1].grid(alpha=0.3, linestyle='--')
            axes[1, 1].plot([0, max(widths)], [0, max(widths)], 'r--', alpha=0.5, label='1:1 ratio')
            axes[1, 1].legend()
            
            plt.tight_layout()
            output_path = os.path.join(output_dir, 'bbox_statistics.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved bbox statistics plot to {output_path}")
        
        except Exception as e:
            logger.error(f"Error plotting bbox statistics: {e}")


class LabelConverter:
    
    @staticmethod
    def coco_to_yolo(coco_json: str, output_dir: str, image_dir: str):
        try:
            logger.info(f"Converting COCO annotations from {coco_json}...")
            
            if not os.path.exists(coco_json):
                raise FileNotFoundError(f"COCO JSON file not found: {coco_json}")
            
            with open(coco_json, 'r') as f:
                coco_data = json.load(f)
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Create image ID to info mapping
            images = {img['id']: img for img in coco_data['images']}
            
            # Group annotations by image
            annotations_by_image = {}
            for ann in coco_data['annotations']:
                img_id = ann['image_id']
                if img_id not in annotations_by_image:
                    annotations_by_image[img_id] = []
                annotations_by_image[img_id].append(ann)
            
            logger.info(f"Converting {len(annotations_by_image)} images...")
            
            # Convert each image's annotations
            for img_id, anns in tqdm(annotations_by_image.items(), desc="Converting"):
                try:
                    img_info = images[img_id]
                    img_width = img_info['width']
                    img_height = img_info['height']
                    
                    label_filename = Path(img_info['file_name']).stem + '.txt'
                    label_path = os.path.join(output_dir, label_filename)
                    
                    with open(label_path, 'w') as f:
                        for ann in anns:
                            # COCO bbox format: [x, y, width, height] (top-left corner)
                            x, y, w, h = ann['bbox']
                            
                            # Convert to YOLO format: [x_center, y_center, width, height] (normalized)
                            x_center = (x + w / 2) / img_width
                            y_center = (y + h / 2) / img_height
                            norm_width = w / img_width
                            norm_height = h / img_height
                            
                            class_id = ann['category_id']
                            
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                
                except KeyError as e:
                    logger.warning(f"Missing key for image {img_id}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error converting image {img_id}: {e}")
                    continue
            
            logger.info(f"Conversion complete. YOLO labels saved to: {output_dir}")
        
        except Exception as e:
            logger.error(f"Error converting COCO to YOLO: {e}")
            raise


def batch_preprocess_images(input_dir: str, output_dir: str,
                           enhance: bool = True, resize: int = None,
                           max_workers: int = None):
    preprocessor = ImagePreprocessor()
    preprocessor.batch_preprocess_parallel(input_dir, output_dir, enhance, resize, max_workers)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLOplan Image Preprocessing Utilities")
    parser.add_argument('--input', type=str, help='Input directory')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--enhance', action='store_true', help='Apply enhancement')
    parser.add_argument('--resize', type=int, default=None, help='Resize target')
    
    args = parser.parse_args()
    
    if args.input and args.output:
        batch_preprocess_images(args.input, args.output, args.enhance, args.resize)
    else:
        parser.print_help()