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
#   Detection:  python YOLOplanDetector.py \
#    --model yolo11s.pt \
#    --source your_electrical_plan.pdf \
#    --output results_test \
#    --dpi 400 \
#    --conf 0.3
# Check results:
#    ls results_test/
# =============================================================================

import os
import sys
import cv2
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from datetime import datetime
import fitz
from pdf2image import convert_from_path
from ultralytics import YOLO
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class YOLOplanDetector:
    
    def __init__(self, model_path: str, conf_threshold: float = 0.25, 
                 iou_threshold: float = 0.45):
        try:
            logger.info(f"Loading YOLO11 model from {model_path}...")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
            self.iou_threshold = iou_threshold
            self.class_names = self.model.names
            logger.info(f"Model loaded successfully. Classes: {self.class_names}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def estimate_image_complexity(self, image: np.ndarray) -> float:
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            return min(edge_density, 1.0)
        except Exception as e:
            logger.warning(f"Error estimating complexity: {e}. Using default.")
            return 0.5
    
    def detect_symbols(self, image_path: str, save_annotated: bool = True,
                      output_dir: str = "results") -> Dict:
        try:
            logger.info(f"Processing image: {image_path}")
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Adaptive confidence threshold
            complexity = self.estimate_image_complexity(img)
            conf_threshold = 0.3 if complexity > 0.5 else self.conf_threshold
            logger.debug(f"Using confidence threshold: {conf_threshold:.2f}")
            
            results = self.model.predict(
                source=image_path,
                conf=conf_threshold,
                iou=self.iou_threshold,
                save=False,
                verbose=False
            )
            
            detections = []
            class_counts = {}
            
            for result in results:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = self.class_names[cls]
                    
                    detection = {
                        'id': f"{os.path.basename(image_path)}_{i}",
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'center': [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                        'area': float((x2 - x1) * (y2 - y1))
                    }
                    detections.append(detection)
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            result_dict = {
                'image_path': image_path,
                'image_name': os.path.basename(image_path),
                'total_detections': len(detections),
                'class_counts': class_counts,
                'detections': detections,
                'timestamp': datetime.now().isoformat()
            }
            
            if save_annotated and detections:
                os.makedirs(output_dir, exist_ok=True)
                annotated_path = self._save_annotated_image(
                    image_path, detections, output_dir
                )
                result_dict['annotated_image'] = annotated_path
            
            # Generate netlist for electrical schematics
            wire_detections = [d for d in detections if 'wire' in d['class'].lower()]
            if wire_detections:
                result_dict['netlist'] = self.generate_netlist(detections, img.shape)
            
            logger.info(f"Found {len(detections)} symbols in {len(class_counts)} classes")
            return result_dict
        
        except Exception as e:
            logger.error(f"Error detecting symbols in {image_path}: {e}")
            raise
    
    def _save_annotated_image(self, image_path: str, detections: List[Dict],
                             output_dir: str) -> str:
        try:
            img = cv2.imread(image_path)
            np.random.seed(42)
            colors = {class_name: tuple(map(int, np.random.randint(0, 255, 3)))
                     for class_name in self.class_names.values()}
            
            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                class_name = det['class']
                conf = det['confidence']
                color = colors.get(class_name, (0, 255, 0))
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name}: {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                cv2.putText(img, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}_annotated.jpg")
            cv2.imwrite(output_path, img)
            logger.info(f"Saved annotated image to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Error saving annotated image: {e}")
            raise
    
    def generate_netlist(self, detections: List[Dict], img_shape: Tuple[int, int, int]) -> Dict:
        try:
            netlist = defaultdict(list)
            wires = [d for d in detections if 'wire' in d['class'].lower()]
            components = [d for d in detections if 'wire' not in d['class'].lower()]
            
            if not wires:
                logger.warning("No wires detected for netlist generation")
                return {}
            
            # Improved connectivity detection
            proximity_threshold = min(img_shape[0], img_shape[1]) * 0.02  # 2% of image dimension
            
            for wire in wires:
                wire_bbox = wire['bbox']
                wire_center = wire['center']
                connected = []
                
                for comp in components:
                    comp_bbox = comp['bbox']
                    comp_center = comp['center']
                    
                    # Check if wire and component are close enough
                    # Multiple checks for better connectivity detection
                    
                    # 1. Check if bboxes overlap or are very close
                    if self._check_proximity(wire_bbox, comp_bbox, proximity_threshold):
                        connected.append(comp['id'])
                        continue
                    
                    # 2. Check if centers are close
                    center_distance = np.sqrt(
                        (wire_center[0] - comp_center[0])**2 + 
                        (wire_center[1] - comp_center[1])**2
                    )
                    if center_distance < proximity_threshold:
                        connected.append(comp['id'])
                
                if connected:
                    netlist[wire['id']] = connected
            
            logger.info(f"Generated netlist with {len(netlist)} wire connections")
            return dict(netlist)
        
        except Exception as e:
            logger.error(f"Error generating netlist: {e}")
            return {}
    
    def _check_proximity(self, bbox1: List[float], bbox2: List[float], threshold: float) -> bool:
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Check if boxes overlap
        if not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min):
            return True
        
        # Check distance between closest edges
        x_dist = max(0, max(x1_min - x2_max, x2_min - x1_max))
        y_dist = max(0, max(y1_min - y2_max, y2_min - y1_max))
        distance = np.sqrt(x_dist**2 + y_dist**2)
        
        return distance < threshold
    
    def process_batch(self, image_paths: List[str], output_dir: str = "results",
                     export_format: str = "csv") -> List[Dict]:
        try:
            os.makedirs(output_dir, exist_ok=True)
            all_results = []
            
            logger.info(f"Processing {len(image_paths)} images...")
            for i, img_path in enumerate(image_paths, 1):
                logger.info(f"[{i}/{len(image_paths)}]")
                result = self.detect_symbols(img_path, save_annotated=True, 
                                            output_dir=output_dir)
                all_results.append(result)
            
            self.export_results(all_results, output_dir, export_format)
            return all_results
        
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise
    
    def export_results(self, results: List[Dict], output_dir: str,
                      format: str = "csv"):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_data = []
            detail_data = []
            netlist_data = []
            
            for result in results:
                summary_row = {
                    'Image': result['image_name'],
                    'Total_Symbols': result['total_detections'],
                    'Timestamp': result['timestamp']
                }
                for class_name, count in result['class_counts'].items():
                    summary_row[f'Count_{class_name}'] = count
                summary_data.append(summary_row)
                
                for det in result['detections']:
                    detail_row = {
                        'Image': result['image_name'],
                        'Class': det['class'],
                        'Confidence': det['confidence'],
                        'X1': det['bbox'][0],
                        'Y1': det['bbox'][1],
                        'X2': det['bbox'][2],
                        'Y2': det['bbox'][3],
                        'Center_X': det['center'][0],
                        'Center_Y': det['center'][1],
                        'Area': det['area'],
                        'ID': det['id']
                    }
                    detail_data.append(detail_row)
                
                if 'netlist' in result and result['netlist']:
                    for wire_id, symbols in result['netlist'].items():
                        netlist_data.append({
                            'Image': result['image_name'],
                            'Wire_ID': wire_id,
                            'Connected_Symbols': ','.join(symbols)
                        })
            
            summary_df = pd.DataFrame(summary_data)
            detail_df = pd.DataFrame(detail_data)
            netlist_df = pd.DataFrame(netlist_data)
            
            if format in ['csv', 'all']:
                summary_path = os.path.join(output_dir, f"summary_{timestamp}.csv")
                detail_path = os.path.join(output_dir, f"details_{timestamp}.csv")
                summary_df.to_csv(summary_path, index=False)
                detail_df.to_csv(detail_path, index=False)
                logger.info(f"Exported CSV files:")
                logger.info(f"  - {summary_path}")
                logger.info(f"  - {detail_path}")
                
                if not netlist_df.empty:
                    netlist_path = os.path.join(output_dir, f"netlist_{timestamp}.csv")
                    netlist_df.to_csv(netlist_path, index=False)
                    logger.info(f"  - {netlist_path}")
            
            if format in ['excel', 'all']:
                excel_path = os.path.join(output_dir, f"report_{timestamp}.xlsx")
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                    detail_df.to_excel(writer, sheet_name='Details', index=False)
                    if not netlist_df.empty:
                        netlist_df.to_excel(writer, sheet_name='Netlist', index=False)
                logger.info(f"Exported Excel file: {excel_path}")
            
            if format in ['json', 'all']:
                json_path = os.path.join(output_dir, f"results_{timestamp}.json")
                with open(json_path, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Exported JSON file: {json_path}")
        
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            raise


class PDFProcessor:
    
    @staticmethod
    def is_vector_pdf(doc: fitz.Document) -> bool:
        try:
            for page in doc:
                if page.get_drawings():
                    return True
            return False
        except Exception as e:
            logger.warning(f"Error checking PDF type: {e}")
            return False
    
    @staticmethod
    def pdf_to_images(pdf_path: str, output_dir: str = "temp_images",
                     dpi: int = 300) -> List[str]:
        try:
            logger.info(f"Converting PDF to images: {pdf_path}")
            os.makedirs(output_dir, exist_ok=True)
            
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
            try:
                images = convert_from_path(pdf_path, dpi=dpi)
                image_paths = []
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                
                for i, image in enumerate(images, 1):
                    img_path = os.path.join(output_dir, f"{base_name}_page_{i}.jpg")
                    image.save(img_path, 'JPEG')
                    image_paths.append(img_path)
                    logger.info(f"  Saved page {i}: {img_path}")
                
                return image_paths
                
            except Exception as e:
                logger.warning(f"pdf2image failed, trying PyMuPDF: {e}")
                doc = fitz.open(pdf_path)
                is_vector = PDFProcessor.is_vector_pdf(doc)
                adaptive_dpi = 600 if is_vector else dpi
                
                logger.info(f"Using DPI: {adaptive_dpi} (vector: {is_vector})")
                
                image_paths = []
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    zoom = adaptive_dpi / 72
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    img_path = os.path.join(output_dir, 
                                           f"{base_name}_page_{page_num + 1}.jpg")
                    pix.save(img_path)
                    image_paths.append(img_path)
                    logger.info(f"  Saved page {page_num + 1}: {img_path}")
                
                doc.close()
                return image_paths
        
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="YOLOplan - Automated Symbol Detection for Technical Drawings"
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to YOLO11 model weights (.pt file)')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image, directory of images, or PDF file')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IOU threshold for NMS (default: 0.45)')
    parser.add_argument('--export', type=str, default='all',
                       choices=['csv', 'excel', 'json', 'all'],
                       help='Export format (default: all)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for PDF conversion (default: 300)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        detector = YOLOplanDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
        
        image_paths = []
        source_path = Path(args.source)
        
        if source_path.is_file():
            if source_path.suffix.lower() == '.pdf':
                logger.info("Detected PDF file. Converting to images...")
                temp_dir = os.path.join(args.output, "temp_pdf_images")
                image_paths = PDFProcessor.pdf_to_images(
                    str(source_path), temp_dir, args.dpi
                )
            elif source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                image_paths = [str(source_path)]
            else:
                logger.error(f"Unsupported file format: {source_path.suffix}")
                sys.exit(1)
        
        elif source_path.is_dir():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.PNG']:
                image_paths.extend([str(p) for p in source_path.glob(ext)])
            if not image_paths:
                logger.error(f"No image files found in {source_path}")
                sys.exit(1)
        
        else:
            logger.error(f"Invalid source path: {args.source}")
            sys.exit(1)
        
        logger.info(f"Found {len(image_paths)} image(s) to process")
        results = detector.process_batch(
            image_paths=image_paths,
            output_dir=args.output,
            export_format=args.export
        )
        
        print("\n" + "="*60)
        print("DETECTION SUMMARY")
        print("="*60)
        total_symbols = sum(r['total_detections'] for r in results)
        all_classes = {}
        for result in results:
            for class_name, count in result['class_counts'].items():
                all_classes[class_name] = all_classes.get(class_name, 0) + count
        
        print(f"Total images processed: {len(results)}")
        print(f"Total symbols detected: {total_symbols}")
        print("\nSymbol counts by class:")
        for class_name, count in sorted(all_classes.items()):
            print(f"  {class_name}: {count}")
        
        print(f"\nResults saved to: {args.output}")
        print("="*60)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()