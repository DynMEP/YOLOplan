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
# Quick Start:
#   Detection:  python YOLOplanDetector.py --model yolo11s.pt --source image.jpg
#   Training:   python YOLOplanTrainer.py train --data data.yaml --model s
#   Setup:      python YOLOplanTrainer.py setup --path datasets/my_data --classes c1 c2
# =============================================================================

import os
import yaml
import shutil
import logging
import argparse
from pathlib import Path
from typing import List, Dict
from ultralytics import YOLO
import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YOLOplanTrainer:
    
    def __init__(self, data_yaml: str, model_size: str = 'n'):
        self.data_yaml = data_yaml
        self.model_size = model_size
        self.model = None
        
        if not os.path.exists(data_yaml):
            raise FileNotFoundError(f"Data YAML not found: {data_yaml}")
        
        try:
            with open(data_yaml, 'r') as f:
                self.data_config = yaml.safe_load(f)
            
            self._validate_dataset()
        except Exception as e:
            logger.error(f"Error loading data config: {e}")
            raise
        
    def _validate_dataset(self):
        logger.info("Validating dataset...")
        required_keys = ['train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in self.data_config:
                raise ValueError(f"Missing required key in data.yaml: {key}")
        
        base_path = Path(self.data_config.get('path', '.'))
        train_path = base_path / self.data_config['train']
        val_path = base_path / self.data_config['val']
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training path not found: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"Validation path not found: {val_path}")
        
        logger.info("✓ Dataset validated")
        logger.info(f"  - Number of classes: {self.data_config['nc']}")
        logger.info(f"  - Classes: {self.data_config['names']}")
        logger.info(f"  - Train path: {train_path}")
        logger.info(f"  - Val path: {val_path}")
    
    def compute_class_weights(self, labels_dir: str) -> List[float]:
        try:
            logger.info("Computing class weights...")
            label_files = list(Path(labels_dir).glob('*.txt'))
            
            if not label_files:
                logger.warning(f"No label files found in {labels_dir}")
                return [1.0] * self.data_config['nc']
            
            class_counts = {i: 0 for i in range(self.data_config['nc'])}
            
            for label_file in label_files:
                try:
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                cls = int(parts[0])
                                if cls < self.data_config['nc']:
                                    class_counts[cls] += 1
                except Exception as e:
                    logger.warning(f"Error reading {label_file}: {e}")
                    continue
            
            total = sum(class_counts.values())
            if total == 0:
                logger.warning("No valid labels found. Using uniform weights.")
                return [1.0] * self.data_config['nc']
            
            # Calculate weights: inverse frequency normalized
            weights = []
            for i in range(self.data_config['nc']):
                count = class_counts.get(i, 1)  # Avoid division by zero
                weight = total / (self.data_config['nc'] * count)
                weights.append(weight)
            
            logger.info(f"Class weights computed: {[f'{w:.3f}' for w in weights]}")
            return weights
        
        except Exception as e:
            logger.error(f"Error computing class weights: {e}")
            return [1.0] * self.data_config['nc']
    
    def optimize_hyperparameters(self, trials: int = 50, epochs_per_trial: int = 10) -> Dict:
        logger.info(f"Starting hyperparameter optimization with {trials} trials...")
        
        def objective(trial):
            try:
                params = {
                    'lr0': trial.suggest_float('lr0', 1e-4, 1e-2, log=True),
                    'momentum': trial.suggest_float('momentum', 0.8, 0.99),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
                    'hsv_h': trial.suggest_float('hsv_h', 0.0, 0.05),
                    'hsv_s': trial.suggest_float('hsv_s', 0.3, 0.9),
                    'hsv_v': trial.suggest_float('hsv_v', 0.2, 0.6),
                    'mosaic': trial.suggest_float('mosaic', 0.0, 1.0),
                    'mixup': trial.suggest_float('mixup', 0.0, 0.3),
                }
                
                model = YOLO(f'yolo11{self.model_size}.pt')
                results = model.train(
                    data=self.data_yaml,
                    epochs=epochs_per_trial,
                    verbose=False,
                    device='0' if torch.cuda.is_available() else 'cpu',
                    **params
                )
                
                # Return validation mAP50 as objective
                metrics = results.results_dict
                return metrics.get('metrics/mAP50(B)', 0.0)
            
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0
        
        try:
            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=42),
                pruner=MedianPruner()
            )
            study.optimize(objective, n_trials=trials, show_progress_bar=True)
            
            logger.info(f"Best trial: {study.best_trial.number}")
            logger.info(f"Best mAP50: {study.best_value:.4f}")
            logger.info(f"Best params: {study.best_params}")
            
            return study.best_params
        
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {e}")
            return {}
    
    def fine_tune(self, new_data_yaml: str, weights: str, epochs: int = 50,
                  freeze_layers: int = 10) -> object:
        try:
            logger.info(f"Fine-tuning model from {weights}...")
            
            if not os.path.exists(weights):
                raise FileNotFoundError(f"Weights file not found: {weights}")
            
            if not os.path.exists(new_data_yaml):
                raise FileNotFoundError(f"Data YAML not found: {new_data_yaml}")
            
            self.model = YOLO(weights)
            results = self.model.train(
                data=new_data_yaml,
                epochs=epochs,
                freeze=freeze_layers,
                imgsz=640,
                device='0' if torch.cuda.is_available() else 'cpu',
                verbose=True
            )
            
            logger.info(f"Fine-tuning complete. Best model: {self.model.trainer.best}")
            return results
        
        except Exception as e:
            logger.error(f"Error in fine-tuning: {e}")
            raise
    
    def train(self, epochs: int = 100, batch: int = 16, imgsz: int = 640,
             patience: int = 50, save_period: int = 10, device: str = None,
             pretrained: bool = True, project: str = 'runs/train',
             name: str = 'yoloplan', exist_ok: bool = False,
             optimize: bool = False, **kwargs):
        try:
            logger.info("="*60)
            logger.info(f"Training YOLO11{self.model_size} on custom symbols")
            logger.info("="*60)
            
            # Initialize model
            if pretrained:
                model_name = f'yolo11{self.model_size}.pt'
                logger.info(f"Loading pretrained model: {model_name}")
                self.model = YOLO(model_name)
            else:
                model_name = f'yolo11{self.model_size}.yaml'
                logger.info(f"Initializing model from scratch: {model_name}")
                self.model = YOLO(model_name)
            
            if device is None:
                device = '0' if torch.cuda.is_available() else 'cpu'
            
            logger.info(f"Using device: {device}")
            
            # Base training arguments
            train_args = {
                'data': self.data_yaml,
                'epochs': epochs,
                'batch': batch,
                'imgsz': imgsz,
                'patience': patience,
                'save_period': save_period,
                'device': device,
                'project': project,
                'name': name,
                'exist_ok': exist_ok,
                'pretrained': pretrained,
                'optimizer': 'auto',
                'verbose': True,
                'seed': 0,
                'deterministic': True,
                'single_cls': False,
                'rect': False,
                'cos_lr': True,
                'close_mosaic': 10,
                'resume': False,
                'amp': True,
                'fraction': 1.0,
                'profile': False,
                'freeze': None,
                'lr0': 0.01,
                'lrf': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
                'warmup_epochs': 3.0,
                'warmup_momentum': 0.8,
                'warmup_bias_lr': 0.1,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'pose': 12.0,
                'kobj': 1.0,
                'label_smoothing': 0.0,
                'nbs': 64,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'mosaic': 1.0,
                'mixup': 0.0,
                'copy_paste': 0.0,
                'auto_augment': 'randaugment',
                'erasing': 0.4,
                'crop_fraction': 1.0,
            }
            
            # Run hyperparameter optimization if requested
            if optimize:
                logger.info("Running hyperparameter optimization...")
                best_params = self.optimize_hyperparameters(trials=50, epochs_per_trial=10)
                if best_params:
                    train_args.update(best_params)
                    logger.info("Applied optimized hyperparameters")
            
            # Update with user-provided kwargs
            train_args.update(kwargs)
            
            logger.info("Starting training...")
            results = self.model.train(**train_args)
            
            logger.info("="*60)
            logger.info("Training completed!")
            logger.info("="*60)
            logger.info(f"Best model saved to: {self.model.trainer.best}")
            logger.info(f"Last model saved to: {self.model.trainer.last}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def validate(self, weights: str = None, split: str = 'val',
                batch: int = 16, imgsz: int = 640, device: str = None):
        try:
            if weights:
                logger.info(f"Loading weights: {weights}")
                if not os.path.exists(weights):
                    raise FileNotFoundError(f"Weights file not found: {weights}")
                self.model = YOLO(weights)
            elif self.model is None:
                raise ValueError("No model loaded. Provide weights or train first.")
            
            if device is None:
                device = '0' if torch.cuda.is_available() else 'cpu'
            
            logger.info(f"Validating on {split} set...")
            results = self.model.val(
                data=self.data_yaml,
                split=split,
                batch=batch,
                imgsz=imgsz,
                device=device
            )
            
            logger.info("Validation complete")
            return results
        
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            raise
    
    def export_model(self, weights: str, format: str = 'onnx',
                    imgsz: int = 640, half: bool = False,
                    simplify: bool = True):
        try:
            logger.info(f"Exporting model to {format}...")
            
            if not os.path.exists(weights):
                raise FileNotFoundError(f"Weights file not found: {weights}")
            
            model = YOLO(weights)
            export_path = model.export(
                format=format,
                imgsz=imgsz,
                half=half,
                simplify=simplify if format == 'onnx' else False
            )
            
            logger.info(f"Model exported to: {export_path}")
            return export_path
        
        except Exception as e:
            logger.error(f"Error exporting model: {e}")
            raise
    
    def auto_generate_yaml(self, labels_dir: str, dataset_path: str,
                          class_names: List[str] = None):
        try:
            logger.info("Auto-generating data.yaml...")
            
            if not os.path.exists(labels_dir):
                raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
            
            # Auto-detect classes if not provided
            if class_names is None:
                class_ids = set()
                for label_file in Path(labels_dir).glob('*.txt'):
                    try:
                        with open(label_file, 'r') as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts:
                                    class_ids.add(int(parts[0]))
                    except Exception as e:
                        logger.warning(f"Error reading {label_file}: {e}")
                        continue
                
                # Generate generic class names
                class_names = [f'class_{i}' for i in sorted(class_ids)]
                logger.warning("Using auto-generated class names. Please update with actual names.")
            
            output_path = os.path.join(dataset_path, 'data.yaml')
            create_dataset_yaml(dataset_path, class_names, output_path)
            
            logger.info(f"Generated data.yaml with {len(class_names)} classes")
        
        except Exception as e:
            logger.error(f"Error auto-generating YAML: {e}")
            raise


def create_dataset_yaml(dataset_path: str, class_names: list,
                       output_path: str = 'data.yaml'):
    try:
        data_config = {
            'path': dataset_path,
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(class_names),
            'names': class_names
        }
        
        with open(output_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Created dataset YAML: {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error creating dataset YAML: {e}")
        raise


def setup_dataset_structure(base_path: str):
    try:
        dirs = [
            'images/train',
            'images/val',
            'images/test',
            'labels/train',
            'labels/val',
            'labels/test'
        ]
        
        base = Path(base_path)
        for dir_path in dirs:
            full_path = base / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created dataset structure at: {base_path}")
        print("\nRecommended structure:")
        print(f"{base_path}/")
        print("├── images/")
        print("│   ├── train/")
        print("│   ├── val/")
        print("│   └── test/")
        print("└── labels/")
        print("    ├── train/")
        print("    ├── val/")
        print("    └── test/")
    
    except Exception as e:
        logger.error(f"Error setting up dataset structure: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="YOLOplan Training - Train YOLO11 for symbol detection"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--data', type=str, required=True,
                            help='Path to data.yaml')
    train_parser.add_argument('--model', type=str, default='n',
                            choices=['n', 's', 'm', 'l', 'x'],
                            help='Model size (default: n)')
    train_parser.add_argument('--epochs', type=int, default=100,
                            help='Number of epochs (default: 100)')
    train_parser.add_argument('--batch', type=int, default=16,
                            help='Batch size (default: 16)')
    train_parser.add_argument('--imgsz', type=int, default=640,
                            help='Image size (default: 640)')
    train_parser.add_argument('--device', type=str, default=None,
                            help='Device (default: auto)')
    train_parser.add_argument('--project', type=str, default='runs/train',
                            help='Project directory')
    train_parser.add_argument('--name', type=str, default='yoloplan',
                            help='Experiment name')
    train_parser.add_argument('--pretrained', action='store_true', default=True,
                            help='Use pretrained weights')
    train_parser.add_argument('--patience', type=int, default=50,
                            help='Early stopping patience')
    train_parser.add_argument('--optimize', action='store_true',
                            help='Run hyperparameter optimization')
    train_parser.add_argument('--weights', type=str, default=None,
                            help='Path to pretrained weights for fine-tuning')
    
    # Validation command
    val_parser = subparsers.add_parser('val', help='Validate model')
    val_parser.add_argument('--weights', type=str, required=True,
                          help='Path to model weights')
    val_parser.add_argument('--data', type=str, required=True,
                          help='Path to data.yaml')
    val_parser.add_argument('--batch', type=int, default=16,
                          help='Batch size')
    val_parser.add_argument('--imgsz', type=int, default=640,
                          help='Image size')
    val_parser.add_argument('--device', type=str, default=None,
                          help='Device (default: auto)')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export model')
    export_parser.add_argument('--weights', type=str, required=True,
                             help='Path to model weights')
    export_parser.add_argument('--format', type=str, default='onnx',
                             choices=['onnx', 'torchscript', 'coreml', 
                                     'engine', 'tflite', 'pb', 'saved_model'],
                             help='Export format')
    export_parser.add_argument('--imgsz', type=int, default=640,
                             help='Image size')
    export_parser.add_argument('--half', action='store_true',
                             help='Use FP16 half precision')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup dataset structure')
    setup_parser.add_argument('--path', type=str, required=True,
                            help='Base path for dataset')
    setup_parser.add_argument('--classes', type=str, nargs='+', required=True,
                            help='List of class names')
    
    args = parser.parse_args()
    
    try:
        if args.command == 'train':
            trainer = YOLOplanTrainer(
                data_yaml=args.data,
                model_size=args.model
            )
            
            # Check if fine-tuning
            if args.weights:
                logger.info("Fine-tuning mode detected")
                trainer.fine_tune(
                    new_data_yaml=args.data,
                    weights=args.weights,
                    epochs=args.epochs
                )
            else:
                trainer.train(
                    epochs=args.epochs,
                    batch=args.batch,
                    imgsz=args.imgsz,
                    device=args.device,
                    project=args.project,
                    name=args.name,
                    pretrained=args.pretrained,
                    patience=args.patience,
                    optimize=args.optimize
                )
        
        elif args.command == 'val':
            trainer = YOLOplanTrainer(data_yaml=args.data)
            trainer.validate(
                weights=args.weights,
                batch=args.batch,
                imgsz=args.imgsz,
                device=args.device
            )
        
        elif args.command == 'export':
            # Create dummy trainer just for export
            trainer = YOLOplanTrainer(data_yaml='dummy.yaml') if os.path.exists('dummy.yaml') else None
            if trainer:
                trainer.export_model(
                    weights=args.weights,
                    format=args.format,
                    imgsz=args.imgsz,
                    half=args.half
                )
            else:
                # Direct export without trainer
                model = YOLO(args.weights)
                export_path = model.export(
                    format=args.format,
                    imgsz=args.imgsz,
                    half=args.half
                )
                logger.info(f"Model exported to: {export_path}")
        
        elif args.command == 'setup':
            setup_dataset_structure(args.path)
            yaml_path = create_dataset_yaml(
                dataset_path=args.path,
                class_names=args.classes,
                output_path=os.path.join(args.path, 'data.yaml')
            )
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()