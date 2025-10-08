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
# Notes:
#   - Requires labeled training data (100-200+ images per class recommended)
#   - GPU recommended for training; CPU acceptable for inference
#   - PDF processing requires Poppler installation
#   - Supports batch processing, multi-format export (CSV/Excel/JSON)
#   - Includes adaptive preprocessing, synthetic data generation, netlist
#   - See INSTALLATION_GUIDE.md for setup and QUICK_REFERENCE.md for commands
# Features:
#   - YOLO11 Integration: Latest YOLO architecture for state-of-the-art detection
#   - Multi-format Support: PDF, JPG, PNG, BMP, TIFF with adaptive DPI
#   - Batch Processing: Parallel processing with progress bars
#   - Export Options: CSV, Excel, JSON with netlist connectivity analysis
#   - Training Pipeline: Complete workflow with hyperparameter optimization
#   - Dataset Tools: Analysis, splitting, augmentation, synthetic generation
#   - Annotated Outputs: Visual results with bounding boxes and labels
#   - Netlist Generation: Connectivity analysis for electrical schematics
# Quick Start:
#   Detection:  python YOLOplanDetector.py --model yolo11s.pt --source image.jpg
#   Training:   python YOLOplanTrainer.py train --data data.yaml --model s
#   Setup:      python YOLOplanTrainer.py setup --path datasets/my_data --classes c1 c2
# =============================================================================

import os
import sys
import pytest
import tempfile
import shutil
import numpy as np
import cv2
from pathlib import Path
import yaml

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ImagePreprocessor import (
    ImagePreprocessor, DatasetSplitter, DatasetAnalyzer, LabelConverter
)


class TestImagePreprocessor:
    
    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_image(self, temp_dir):
        img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        img_path = os.path.join(temp_dir, "test_image.jpg")
        cv2.imwrite(img_path, img)
        return img_path
    
    def test_estimate_noise(self):
        preprocessor = ImagePreprocessor()
        img = np.random.randint(0, 255, (640, 640), dtype=np.uint8)
        noise = preprocessor.estimate_noise(img)
        assert isinstance(noise, float)
        assert noise >= 0.0
    
    def test_analyze_contrast(self):
        preprocessor = ImagePreprocessor()
        img = np.random.randint(0, 255, (640, 640), dtype=np.uint8)
        contrast = preprocessor.analyze_contrast(img)
        assert isinstance(contrast, float)
        assert contrast >= 0.0
    
    def test_enhance_drawing(self, sample_image, temp_dir):
        preprocessor = ImagePreprocessor()
        output_path = os.path.join(temp_dir, "enhanced.jpg")
        
        result = preprocessor.enhance_drawing(
            sample_image, output_path, 
            sharpen=True, denoise=True, contrast=True
        )
        
        assert os.path.exists(output_path)
        assert result is not None
        assert result.shape[2] == 3  # BGR image
    
    def test_resize_maintain_aspect(self, sample_image, temp_dir):
        preprocessor = ImagePreprocessor()
        output_path = os.path.join(temp_dir, "resized.jpg")
        
        result = preprocessor.resize_maintain_aspect(
            sample_image, target_size=320, output_path=output_path
        )
        
        assert os.path.exists(output_path)
        assert result is not None
        assert max(result.shape[:2]) == 320
    
    def test_remove_background(self, sample_image, temp_dir):
        preprocessor = ImagePreprocessor()
        output_path = os.path.join(temp_dir, "no_bg.jpg")
        
        result = preprocessor.remove_background(
            sample_image, output_path, threshold=240
        )
        
        assert os.path.exists(output_path)
        assert result is not None
    
    def test_generate_synthetic_schematic(self, temp_dir):
        preprocessor = ImagePreprocessor()
        class_names = ['outlet', 'switch', 'light']
        
        preprocessor.generate_synthetic_schematic(
            class_names=class_names,
            output_dir=temp_dir,
            num_images=5,
            img_size=640
        )
        
        images = list(Path(temp_dir).glob('*.jpg'))
        labels = list(Path(temp_dir).glob('*.txt'))
        
        assert len(images) == 5
        assert len(labels) == 5


class TestDatasetSplitter:
    
    @pytest.fixture
    def temp_dataset(self):
        temp_dir = tempfile.mkdtemp()
        images_dir = os.path.join(temp_dir, 'images')
        labels_dir = os.path.join(temp_dir, 'labels')
        os.makedirs(images_dir)
        os.makedirs(labels_dir)
        
        # Create 10 test images and labels
        for i in range(10):
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img_path = os.path.join(images_dir, f"img_{i:03d}.jpg")
            cv2.imwrite(img_path, img)
            
            label_path = os.path.join(labels_dir, f"img_{i:03d}.txt")
            with open(label_path, 'w') as f:
                f.write(f"0 0.5 0.5 0.1 0.1\n")
        
        yield temp_dir, images_dir, labels_dir
        shutil.rmtree(temp_dir)
    
    def test_split_dataset(self, temp_dataset):
        temp_dir, images_dir, labels_dir = temp_dataset
        output_dir = os.path.join(temp_dir, 'split')
        
        splitter = DatasetSplitter()
        splitter.split_dataset(
            images_dir=images_dir,
            labels_dir=labels_dir,
            output_dir=output_dir,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1
        )
        
        # Check that directories were created
        assert os.path.exists(os.path.join(output_dir, 'images', 'train'))
        assert os.path.exists(os.path.join(output_dir, 'images', 'val'))
        assert os.path.exists(os.path.join(output_dir, 'images', 'test'))
        assert os.path.exists(os.path.join(output_dir, 'labels', 'train'))
        assert os.path.exists(os.path.join(output_dir, 'labels', 'val'))
        assert os.path.exists(os.path.join(output_dir, 'labels', 'test'))
        
        # Check that files were distributed
        train_imgs = list(Path(output_dir, 'images', 'train').glob('*.jpg'))
        val_imgs = list(Path(output_dir, 'images', 'val').glob('*.jpg'))
        test_imgs = list(Path(output_dir, 'images', 'test').glob('*.jpg'))
        
        assert len(train_imgs) + len(val_imgs) + len(test_imgs) == 10
        assert len(train_imgs) == 7  # 70% of 10
        assert len(val_imgs) == 2    # 20% of 10
        assert len(test_imgs) == 1   # 10% of 10
    
    def test_invalid_ratios(self, temp_dataset):
        temp_dir, images_dir, labels_dir = temp_dataset
        output_dir = os.path.join(temp_dir, 'split')
        
        splitter = DatasetSplitter()
        with pytest.raises(ValueError):
            splitter.split_dataset(
                images_dir=images_dir,
                labels_dir=labels_dir,
                output_dir=output_dir,
                train_ratio=0.6,
                val_ratio=0.3,
                test_ratio=0.2  # Sum > 1.0
            )


class TestDatasetAnalyzer:
    
    @pytest.fixture
    def temp_labels(self):
        temp_dir = tempfile.mkdtemp()
        
        # Create 5 label files with varying numbers of objects
        for i in range(5):
            label_path = os.path.join(temp_dir, f"label_{i}.txt")
            with open(label_path, 'w') as f:
                # Write 1-5 random labels per file
                for j in range(i + 1):
                    cls = j % 3  # 3 classes
                    x = np.random.uniform(0.2, 0.8)
                    y = np.random.uniform(0.2, 0.8)
                    w = np.random.uniform(0.05, 0.2)
                    h = np.random.uniform(0.05, 0.2)
                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_analyze_dataset(self, temp_labels):
        output_dir = os.path.join(temp_labels, 'analysis')
        class_names = ['outlet', 'switch', 'light']
        
        analyzer = DatasetAnalyzer()
        analyzer.analyze_dataset(
            labels_dir=temp_labels,
            class_names=class_names,
            output_dir=output_dir
        )
        
        # Check that analysis files were created
        assert os.path.exists(os.path.join(output_dir, 'class_distribution.png'))
        assert os.path.exists(os.path.join(output_dir, 'bbox_statistics.png'))
        assert os.path.exists(os.path.join(output_dir, 'dataset_stats.txt'))


class TestLabelConverter:
    
    @pytest.fixture
    def temp_coco_json(self):
        temp_dir = tempfile.mkdtemp()
        
        coco_data = {
            'images': [
                {
                    'id': 1,
                    'file_name': 'image1.jpg',
                    'width': 640,
                    'height': 480
                },
                {
                    'id': 2,
                    'file_name': 'image2.jpg',
                    'width': 800,
                    'height': 600
                }
            ],
            'annotations': [
                {
                    'id': 1,
                    'image_id': 1,
                    'category_id': 0,
                    'bbox': [100, 100, 50, 50]  # x, y, w, h
                },
                {
                    'id': 2,
                    'image_id': 1,
                    'category_id': 1,
                    'bbox': [200, 150, 60, 40]
                },
                {
                    'id': 3,
                    'image_id': 2,
                    'category_id': 0,
                    'bbox': [150, 200, 80, 70]
                }
            ],
            'categories': [
                {'id': 0, 'name': 'outlet'},
                {'id': 1, 'name': 'switch'}
            ]
        }
        
        json_path = os.path.join(temp_dir, 'annotations.json')
        import json
        with open(json_path, 'w') as f:
            json.dump(coco_data, f)
        
        yield temp_dir, json_path
        shutil.rmtree(temp_dir)
    
    def test_coco_to_yolo(self, temp_coco_json):
        temp_dir, json_path = temp_coco_json
        output_dir = os.path.join(temp_dir, 'yolo_labels')
        
        converter = LabelConverter()
        converter.coco_to_yolo(
            coco_json=json_path,
            output_dir=output_dir,
            image_dir=temp_dir
        )
        
        # Check that YOLO label files were created
        assert os.path.exists(os.path.join(output_dir, 'image1.txt'))
        assert os.path.exists(os.path.join(output_dir, 'image2.txt'))
        
        # Verify content of first label file
        with open(os.path.join(output_dir, 'image1.txt'), 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2  # Two annotations for image1
            
            # Check first annotation
            parts = lines[0].strip().split()
            assert len(parts) == 5
            assert int(parts[0]) == 0  # class_id
            
            # Verify coordinates are normalized (0-1)
            for val in parts[1:]:
                assert 0.0 <= float(val) <= 1.0


class TestYOLOplanDetector:
    
    def test_estimate_image_complexity(self):
        from YOLOplanDetector import YOLOplanDetector
        
        # Mock detector without loading actual model
        detector = type('obj', (object,), {
            'estimate_image_complexity': YOLOplanDetector.estimate_image_complexity
        })()
        
        # Simple image
        simple_img = np.ones((640, 640, 3), dtype=np.uint8) * 128
        complexity_simple = detector.estimate_image_complexity(detector, simple_img)
        
        # Complex image with edges
        complex_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        complexity_complex = detector.estimate_image_complexity(detector, complex_img)
        
        assert 0.0 <= complexity_simple <= 1.0
        assert 0.0 <= complexity_complex <= 1.0
        assert complexity_complex > complexity_simple
    
    def test_check_proximity(self):
        from YOLOplanDetector import YOLOplanDetector
        
        detector = type('obj', (object,), {
            '_check_proximity': YOLOplanDetector._check_proximity
        })()
        
        # Overlapping boxes
        bbox1 = [100, 100, 200, 200]
        bbox2 = [150, 150, 250, 250]
        assert detector._check_proximity(detector, bbox1, bbox2, threshold=10)
        
        # Close boxes
        bbox3 = [100, 100, 150, 150]
        bbox4 = [155, 155, 200, 200]
        assert detector._check_proximity(detector, bbox3, bbox4, threshold=10)
        
        # Far boxes
        bbox5 = [100, 100, 150, 150]
        bbox6 = [300, 300, 350, 350]
        assert not detector._check_proximity(detector, bbox5, bbox6, threshold=10)


class TestYOLOplanTrainer:
    
    @pytest.fixture
    def temp_dataset_yaml(self):
        temp_dir = tempfile.mkdtemp()
        
        # Create directory structure
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(temp_dir, 'images', split))
            os.makedirs(os.path.join(temp_dir, 'labels', split))
        
        # Create YAML file
        yaml_path = os.path.join(temp_dir, 'data.yaml')
        data_config = {
            'path': temp_dir,
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 3,
            'names': ['outlet', 'switch', 'light']
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f)
        
        yield temp_dir, yaml_path
        shutil.rmtree(temp_dir)
    
    def test_trainer_initialization(self, temp_dataset_yaml):
        from YOLOplanTrainer import YOLOplanTrainer
        
        temp_dir, yaml_path = temp_dataset_yaml
        
        trainer = YOLOplanTrainer(
            data_yaml=yaml_path,
            model_size='n'
        )
        
        assert trainer.data_yaml == yaml_path
        assert trainer.model_size == 'n'
        assert trainer.data_config['nc'] == 3
        assert len(trainer.data_config['names']) == 3
    
    def test_compute_class_weights(self, temp_dataset_yaml):
        from YOLOplanTrainer import YOLOplanTrainer
        
        temp_dir, yaml_path = temp_dataset_yaml
        labels_dir = os.path.join(temp_dir, 'labels', 'train')
        
        # Create some label files
        for i in range(5):
            label_path = os.path.join(labels_dir, f"label_{i}.txt")
            with open(label_path, 'w') as f:
                # Imbalanced classes: mostly class 0
                for j in range(10 if i % 2 == 0 else 2):
                    cls = 0 if j < 8 else 1
                    f.write(f"{cls} 0.5 0.5 0.1 0.1\n")
        
        trainer = YOLOplanTrainer(data_yaml=yaml_path, model_size='n')
        weights = trainer.compute_class_weights(labels_dir)
        
        assert len(weights) == 3  # 3 classes
        assert all(w > 0 for w in weights)
        # Class with fewer samples should have higher weight
        assert weights[1] > weights[0] or weights[2] > weights[0]


class TestPDFProcessor:
    
    def test_is_vector_pdf(self):
        from YOLOplanDetector import PDFProcessor
        import fitz
        
        # Create a simple PDF with text (vector)
        temp_dir = tempfile.mkdtemp()
        pdf_path = os.path.join(temp_dir, 'test.pdf')
        
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((100, 100), "Test text")
        doc.save(pdf_path)
        doc.close()
        
        # Test detection
        doc = fitz.open(pdf_path)
        result = PDFProcessor.is_vector_pdf(doc)
        doc.close()
        
        shutil.rmtree(temp_dir)
        
        # Text might not be detected as vector graphics
        assert isinstance(result, bool)


class TestUtilityFunctions:
    
    def test_create_dataset_yaml(self):
        from YOLOplanTrainer import create_dataset_yaml
        
        temp_dir = tempfile.mkdtemp()
        yaml_path = os.path.join(temp_dir, 'data.yaml')
        
        create_dataset_yaml(
            dataset_path=temp_dir,
            class_names=['outlet', 'switch', 'light'],
            output_path=yaml_path
        )
        
        assert os.path.exists(yaml_path)
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert config['nc'] == 3
        assert config['names'] == ['outlet', 'switch', 'light']
        assert 'train' in config
        assert 'val' in config
        
        shutil.rmtree(temp_dir)
    
    def test_setup_dataset_structure(self):
        from YOLOplanTrainer import setup_dataset_structure
        
        temp_dir = tempfile.mkdtemp()
        
        setup_dataset_structure(temp_dir)
        
        # Check that all directories were created
        for split in ['train', 'val', 'test']:
            assert os.path.exists(os.path.join(temp_dir, 'images', split))
            assert os.path.exists(os.path.join(temp_dir, 'labels', split))
        
        shutil.rmtree(temp_dir)


# Integration tests
class TestIntegration:
    
    @pytest.fixture
    def complete_dataset(self):
        temp_dir = tempfile.mkdtemp()
        
        # Create images and labels
        images_dir = os.path.join(temp_dir, 'raw_images')
        labels_dir = os.path.join(temp_dir, 'raw_labels')
        os.makedirs(images_dir)
        os.makedirs(labels_dir)
        
        for i in range(10):
            # Create image
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img_path = os.path.join(images_dir, f"img_{i:03d}.jpg")
            cv2.imwrite(img_path, img)
            
            # Create label
            label_path = os.path.join(labels_dir, f"img_{i:03d}.txt")
            with open(label_path, 'w') as f:
                for j in range(5):
                    cls = j % 3
                    x = np.random.uniform(0.2, 0.8)
                    y = np.random.uniform(0.2, 0.8)
                    w = np.random.uniform(0.05, 0.2)
                    h = np.random.uniform(0.05, 0.2)
                    f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        
        yield temp_dir, images_dir, labels_dir
        shutil.rmtree(temp_dir)
    
    def test_full_preprocessing_workflow(self, complete_dataset):
        temp_dir, images_dir, labels_dir = complete_dataset
        
        # Step 1: Preprocess images
        preprocessor = ImagePreprocessor()
        processed_dir = os.path.join(temp_dir, 'processed')
        preprocessor.batch_preprocess_parallel(
            input_dir=images_dir,
            output_dir=processed_dir,
            enhance=True,
            resize=640
        )
        
        # Step 2: Split dataset
        splitter = DatasetSplitter()
        split_dir = os.path.join(temp_dir, 'split')
        splitter.split_dataset(
            images_dir=processed_dir,
            labels_dir=labels_dir,
            output_dir=split_dir,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1
        )
        
        # Step 3: Analyze dataset
        analyzer = DatasetAnalyzer()
        analysis_dir = os.path.join(temp_dir, 'analysis')
        analyzer.analyze_dataset(
            labels_dir=os.path.join(split_dir, 'labels', 'train'),
            class_names=['outlet', 'switch', 'light'],
            output_dir=analysis_dir
        )
        
        # Verify all steps completed
        assert os.path.exists(processed_dir)
        assert os.path.exists(split_dir)
        assert os.path.exists(analysis_dir)
        assert os.path.exists(os.path.join(analysis_dir, 'class_distribution.png'))


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])